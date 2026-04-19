"""
Interpretability analyses for the MoE router.

Five analyses:
  1. α distribution by token language (LID gold labels)
  2. α trajectory at code-switch boundaries (±3 token window)
  3. α distribution bucketed by sentence CMI
  4. Expert disagreement (L2) vs α
  5. Cross-task α comparison

All analyses save both PNG figures and a JSON stats file so the
analysis webpage can display numbers without re-running the model.

Public surface:
  extract_alphas(model, dataloader, device, task) -> list[dict]
  run_analyses(model, dataloader, task, device, figures_dir, exp_id) -> dict
  run_cross_task_analyses(records_per_task, figures_dir) -> str (path)
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

import configs
from training import _to_device

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Alpha extraction
# ---------------------------------------------------------------------------

def extract_alphas(model, dataloader, device, task: str) -> list[dict]:
    """Run model on dataloader; return list of per-word dicts.

    Each dict contains:
      sentence_id, position, word, label (str), alpha (float),
      l2_disagreement (float | None)
    """
    id2label    = {i: l for i, l in enumerate(configs.TASK_LABELS[task])}
    model.eval()
    records     = []
    sent_offset = 0

    with torch.no_grad():
        for batch in dataloader:
            batch   = _to_device(batch, device)
            out     = model(batch)
            alpha   = out.get("alpha")
            l2_dist = out.get("l2_disagreement")

            if alpha is None:
                sent_offset += batch["hing_input_ids"].shape[0]
                continue

            alpha   = alpha.cpu()
            l2_dist = l2_dist.cpu() if l2_dist is not None else None
            labels  = batch["labels"].cpu()

            for i in range(alpha.shape[0]):
                n     = batch["num_words"][i].item()
                words = batch["words"][i]
                sid   = sent_offset + i

                for pos in range(n):
                    lab_id = labels[i, pos].item()
                    if lab_id == -100:
                        continue
                    records.append({
                        "sentence_id":     sid,
                        "position":        pos,
                        "word":            words[pos] if pos < len(words) else "",
                        "label":           id2label.get(lab_id, "?"),
                        "alpha":           round(alpha[i, pos].item(), 6),
                        "l2_disagreement": (
                            round(l2_dist[i, pos].item(), 6)
                            if l2_dist is not None else None
                        ),
                    })

            sent_offset += alpha.shape[0]

    print(f"  Extracted {len(records)} word-level records.")
    return records


# ---------------------------------------------------------------------------
# CMI helper
# ---------------------------------------------------------------------------

def _compute_cmi(labels: list[str], task: str) -> float:
    """Gambäck & Das CMI: (N - max_lang_count) / N for LID tokens only."""
    if task != "lid":
        return 0.0
    tokens = [l for l in labels if l != "ot"]
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    N      = len(tokens)
    return (N - max(counts.values())) / N


def _group_by_sentence(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[r["sentence_id"]].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Stats computation (returns dict for JSON serialization)
# ---------------------------------------------------------------------------

def _compute_stats(records: list[dict], task: str, exp_id: str,
                   prefix: str) -> dict:
    """Compute summary statistics from records; return serialisable dict."""
    alphas_all = [r["alpha"] for r in records]

    # Alpha by label
    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r["alpha"])

    alpha_by_label = {
        lab: {
            "mean":  round(float(np.mean(vals)), 5),
            "std":   round(float(np.std(vals)),  5),
            "count": len(vals),
        }
        for lab, vals in sorted(by_label.items()) if vals
    }

    # L2 correlation
    l2s    = [r["l2_disagreement"] for r in records if r["l2_disagreement"] is not None]
    alp_l2 = [r["alpha"]           for r in records if r["l2_disagreement"] is not None]
    l2_corr = float(np.corrcoef(l2s, alp_l2)[0, 1]) if len(l2s) > 1 else None

    # Switch-point trajectory (±3 window)
    sentences = _group_by_sentence(records)
    window    = 3
    offsets   = list(range(-window, window + 1))
    alpha_by_offset: dict[int, list] = {o: [] for o in offsets}

    for sent_recs in sentences.values():
        sent_recs = sorted(sent_recs, key=lambda r: r["position"])
        lbls  = [r["label"] for r in sent_recs]
        alps  = [r["alpha"] for r in sent_recs]
        for i in range(1, len(lbls)):
            if lbls[i] != lbls[i - 1] and lbls[i] in ("hi", "en") \
                    and lbls[i - 1] in ("hi", "en"):
                for off in offsets:
                    j = i + off
                    if 0 <= j < len(alps):
                        alpha_by_offset[off].append(alps[j])

    switch_trajectory = {
        str(o): {
            "mean": round(float(np.mean(alpha_by_offset[o])), 5)
                    if alpha_by_offset[o] else None,
            "sem":  round(float(np.std(alpha_by_offset[o]) /
                          np.sqrt(len(alpha_by_offset[o]) + 1e-9)), 5)
                    if alpha_by_offset[o] else None,
            "count": len(alpha_by_offset[o]),
        }
        for o in offsets
    }

    # CMI buckets
    n_buckets = 5
    sent_cmis   = {}
    sent_alphas = {}
    for sid, sent_recs in sentences.items():
        sent_recs = sorted(sent_recs, key=lambda r: r["position"])
        sent_cmis[sid]   = _compute_cmi([r["label"] for r in sent_recs], task)
        sent_alphas[sid] = [r["alpha"] for r in sent_recs]

    cmis      = np.array(list(sent_cmis.values()))
    cmi_stats = []
    if cmis.max() > 0:
        edges = np.linspace(0, cmis.max(), n_buckets + 1)
        buckets: list[list] = [[] for _ in range(n_buckets)]
        for sid, cmi in sent_cmis.items():
            b = min(int(cmi / (cmis.max() + 1e-9) * n_buckets), n_buckets - 1)
            buckets[b].extend(sent_alphas[sid])
        cmi_stats = [
            {
                "label":  f"{edges[i]:.2f}–{edges[i+1]:.2f}",
                "mean":   round(float(np.mean(b)), 5) if b else None,
                "median": round(float(np.median(b)), 5) if b else None,
                "q25":    round(float(np.percentile(b, 25)), 5) if b else None,
                "q75":    round(float(np.percentile(b, 75)), 5) if b else None,
                "count":  len(b),
            }
            for i, b in enumerate(buckets)
        ]

    # Figure filenames (relative to figures_dir)
    def _fname(suffix):
        return os.path.basename(prefix + suffix)

    return {
        "exp_id":        exp_id,
        "task":          task,
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "num_records":   len(records),
        "mean_alpha":    round(float(np.mean(alphas_all)), 5),
        "std_alpha":     round(float(np.std(alphas_all)),  5),
        "alpha_by_label": alpha_by_label,
        "l2_correlation": round(l2_corr, 5) if l2_corr is not None else None,
        "switch_trajectory": switch_trajectory,
        "cmi_buckets":   cmi_stats,
        "figures": {
            "alpha_by_lang": _fname("alpha_by_lang.png"),
            "switch_points": _fname("switch_points.png"),
            "cmi_buckets":   _fname("cmi_buckets.png"),
            "disagreement":  _fname("disagreement.png"),
        },
    }


# ---------------------------------------------------------------------------
# Analysis 1: α by token language
# ---------------------------------------------------------------------------

def analysis_alpha_by_language(records: list[dict], save_path: str):
    groups    = {"hi": [], "en": [], "ot": []}
    for r in records:
        if r["label"] in groups:
            groups[r["label"]].append(r["alpha"])

    if not any(groups.values()):
        print("  Analysis 1: no LID-labelled records, skipping.")
        return

    fig, ax   = plt.subplots(figsize=(7, 4))
    label_map = {"hi": "Hindi (hi)", "en": "English (en)", "ot": "Other (ot)"}
    colours   = {"hi": "#e05c5c", "en": "#4a90d9", "ot": "#7dbb7d"}
    for lang, alphas in groups.items():
        if alphas:
            sns.kdeplot(alphas, ax=ax, label=label_map[lang],
                        fill=True, alpha=0.3, color=colours[lang])

    ax.set_xlabel("α  (1 = full HingBERT, 0 = full RoBERTa)")
    ax.set_ylabel("Density")
    ax.set_title("α distribution by token language")
    ax.legend()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Analysis 2: α at code-switch boundaries
# ---------------------------------------------------------------------------

def analysis_switch_points(records: list[dict], save_path: str, window: int = 3):
    sentences       = _group_by_sentence(records)
    offsets         = list(range(-window, window + 1))
    alpha_by_offset = {o: [] for o in offsets}

    for sent_recs in sentences.values():
        sent_recs = sorted(sent_recs, key=lambda r: r["position"])
        labels    = [r["label"] for r in sent_recs]
        alphas    = [r["alpha"] for r in sent_recs]
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1] \
                    and labels[i] in ("hi", "en") \
                    and labels[i - 1] in ("hi", "en"):
                for off in offsets:
                    j = i + off
                    if 0 <= j < len(alphas):
                        alpha_by_offset[off].append(alphas[j])

    means = [np.mean(alpha_by_offset[o]) if alpha_by_offset[o] else np.nan
             for o in offsets]
    sems  = [np.std(alpha_by_offset[o]) / np.sqrt(len(alpha_by_offset[o]) + 1e-9)
             for o in offsets]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(offsets, means, yerr=sems, marker="o", capsize=4, color="#4a90d9")
    ax.axvline(x=-0.5, color="#e05c5c", linestyle="--", alpha=0.6, label="switch boundary")
    ax.set_xlabel("Token offset from switch point")
    ax.set_ylabel("Mean α")
    ax.set_title("α trajectory at code-switch boundaries (±3 tokens)")
    ax.legend()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Analysis 3: α vs sentence CMI
# ---------------------------------------------------------------------------

def analysis_cmi_buckets(records: list[dict], task: str,
                          save_path: str, n_buckets: int = 5):
    sentences   = _group_by_sentence(records)
    sent_cmis   = {}
    sent_alphas = {}
    for sid, sent_recs in sentences.items():
        sent_recs = sorted(sent_recs, key=lambda r: r["position"])
        sent_cmis[sid]   = _compute_cmi([r["label"] for r in sent_recs], task)
        sent_alphas[sid] = [r["alpha"] for r in sent_recs]

    cmis = np.array(list(sent_cmis.values()))
    if cmis.max() == 0:
        print("  Analysis 3: all CMI=0 (non-LID task or monolingual data), skipping.")
        return

    edges  = np.linspace(0, cmis.max(), n_buckets + 1)
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(n_buckets)]
    buckets: list[list] = [[] for _ in range(n_buckets)]
    for sid, cmi in sent_cmis.items():
        b = min(int(cmi / (cmis.max() + 1e-9) * n_buckets), n_buckets - 1)
        buckets[b].extend(sent_alphas[sid])

    data   = [b for b in buckets if b]
    lbls   = [labels[i] for i, b in enumerate(buckets) if b]

    fig, ax = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot(data, labels=lbls, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4a90d940")
        patch.set_edgecolor("#4a90d9")
    ax.set_xlabel("CMI bucket")
    ax.set_ylabel("α")
    ax.set_title("α distribution by sentence CMI")
    plt.xticks(rotation=20)
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Analysis 4: expert disagreement vs α
# ---------------------------------------------------------------------------

def analysis_expert_disagreement(records: list[dict], save_path: str,
                                  max_pts: int = 5000):
    l2s    = [r["l2_disagreement"] for r in records if r["l2_disagreement"] is not None]
    alphas = [r["alpha"]           for r in records if r["l2_disagreement"] is not None]

    if not l2s:
        print("  Analysis 4: no L2 disagreement data, skipping.")
        return

    if len(l2s) > max_pts:
        idx    = np.random.choice(len(l2s), max_pts, replace=False)
        l2s    = [l2s[i]    for i in idx]
        alphas = [alphas[i] for i in idx]

    corr = np.corrcoef(l2s, alphas)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(l2s, alphas, alpha=0.18, s=8, color="#4a90d9", rasterized=True)
    ax.set_xlabel("L2 distance  ‖h_hing − h_rob‖")
    ax.set_ylabel("α")
    ax.set_title(f"Expert disagreement vs α  (r = {corr:.3f})")
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Analysis 5: cross-task α comparison
# ---------------------------------------------------------------------------

def analysis_cross_task(records_per_task: dict, save_path: str):
    """Violin plot comparing α distributions across tasks.

    records_per_task: {task_label: list[dict]}
    """
    task_names = list(records_per_task.keys())
    data = [[r["alpha"] for r in records_per_task[t]] for t in task_names]
    if not any(data):
        print("  Analysis 5: no data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    parts = ax.violinplot(data, positions=range(len(task_names)), showmedians=True)
    for pc in parts.get("bodies", []):
        pc.set_facecolor("#4a90d9")
        pc.set_alpha(0.5)
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels([t.upper() for t in task_names])
    ax.set_ylabel("α")
    ax.set_title("α distribution across tasks")
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_analyses(model, dataloader, task: str, device,
                 figures_dir: str, exp_id: str = "") -> dict:
    """Run all applicable analyses; save PNGs + stats JSON. Returns stats dict."""
    os.makedirs(figures_dir, exist_ok=True)
    prefix = os.path.join(figures_dir,
                          f"{exp_id}_{task}_" if exp_id else f"{task}_")

    print(f"\nExtracting alphas  task={task}  exp={exp_id} ...")
    records = extract_alphas(model, dataloader, device, task)
    if not records:
        print("  No records with alpha — model has no router, skipping analyses.")
        return {}

    # ---- save raw records for later use ---------------------------------
    records_path = prefix + "records.json"
    with open(records_path, "w") as f:
        json.dump(records, f)
    print(f"  Records saved -> {records_path}")

    # ---- run figures ----------------------------------------------------
    print("  Analysis 1: α by language ...")
    analysis_alpha_by_language(records, prefix + "alpha_by_lang.png")

    print("  Analysis 2: switch-point trajectory ...")
    analysis_switch_points(records, prefix + "switch_points.png")

    print("  Analysis 3: α vs CMI ...")
    analysis_cmi_buckets(records, task, prefix + "cmi_buckets.png")

    print("  Analysis 4: expert disagreement ...")
    analysis_expert_disagreement(records, prefix + "disagreement.png")

    # ---- compute & save stats JSON --------------------------------------
    stats = _compute_stats(records, task, exp_id, prefix)
    stats_path = prefix + "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved   -> {stats_path}")

    return stats


def run_cross_task_analyses(records_per_task: dict, figures_dir: str) -> str:
    """Save cross-task violin plot and stats. Returns path of saved figure."""
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "cross_task_alpha.png")
    analysis_cross_task(records_per_task, save_path)

    # Save cross-task stats JSON
    stats = {
        task: {
            "mean_alpha": round(float(np.mean([r["alpha"] for r in recs])), 5),
            "std_alpha":  round(float(np.std([r["alpha"]  for r in recs])), 5),
            "count":      len(recs),
        }
        for task, recs in records_per_task.items() if recs
    }
    stats_path = os.path.join(figures_dir, "cross_task_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Cross-task stats -> {stats_path}")
    return save_path


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------

def _save(fig, path: str):
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {path}")
