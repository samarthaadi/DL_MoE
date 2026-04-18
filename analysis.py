"""
Interpretability analyses for the MoE router.

Five analyses:
  1. α distribution by token language (LID gold labels)
  2. α trajectory at code-switch boundaries (±3 token window)
  3. α distribution bucketed by sentence CMI
  4. Expert disagreement (L2) vs α
  5. Cross-task α comparison

All analyses take a list of per-word record dicts produced by extract_alphas().
"""

import os

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
    id2label = {i: l for i, l in enumerate(configs.TASK_LABELS[task])}
    model.eval()
    records   = []
    sent_offset = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _to_device(batch, device)
            out   = model(batch)

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
                        "sentence_id":    sid,
                        "position":       pos,
                        "word":           words[pos] if pos < len(words) else "",
                        "label":          id2label.get(lab_id, "?"),
                        "alpha":          alpha[i, pos].item(),
                        "l2_disagreement": (
                            l2_dist[i, pos].item() if l2_dist is not None else None
                        ),
                    })

            sent_offset += alpha.shape[0]

    print(f"  Extracted {len(records)} word-level records.")
    return records


# ---------------------------------------------------------------------------
# CMI helper
# ---------------------------------------------------------------------------

def _compute_cmi(labels: list[str], task: str) -> float:
    """Gambäck & Das CMI for one sentence.

    CMI = (N - max_i count_i) / N  where N = #tokens not tagged 'other'/'O'/'X'.
    Returns 0.0 for monolingual or empty sentences.
    """
    if task == "lid":
        tokens = [l for l in labels if l != "ot"]
    else:
        # For NER/POS use LID gold labels if available; fallback: skip
        return 0.0

    if not tokens:
        return 0.0

    from collections import Counter
    counts = Counter(tokens)
    N      = len(tokens)
    max_c  = max(counts.values())
    return (N - max_c) / N


def _group_by_sentence(records: list[dict]) -> dict:
    """Group records by sentence_id."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r["sentence_id"]].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Analysis 1: α by token language
# ---------------------------------------------------------------------------

def analysis_alpha_by_language(records: list[dict], save_path: str):
    """Plot α distribution for Hindi / English / Other tokens."""
    groups = {"hi": [], "en": [], "ot": []}
    for r in records:
        lab = r["label"]
        if lab in groups:
            groups[lab].append(r["alpha"])

    if not any(groups.values()):
        print("  Analysis 1: no LID-labelled records found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    label_map = {"hi": "Hindi (hi)", "en": "English (en)", "ot": "Other (ot)"}
    for lang, alphas in groups.items():
        if alphas:
            sns.kdeplot(alphas, ax=ax, label=label_map[lang], fill=True, alpha=0.3)

    ax.set_xlabel("α  (1 = full HingBERT, 0 = full RoBERTa)")
    ax.set_ylabel("Density")
    ax.set_title("α distribution by token language")
    ax.legend()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Analysis 2: α at code-switch boundaries
# ---------------------------------------------------------------------------

def analysis_switch_points(records: list[dict], save_path: str, window: int = 3):
    """Plot mean α trajectory over a ±window token window around language switches."""
    sentences = _group_by_sentence(records)
    offsets   = list(range(-window, window + 1))
    alpha_by_offset = {o: [] for o in offsets}

    for sid, sent_recs in sentences.items():
        sent_recs = sorted(sent_recs, key=lambda r: r["position"])
        labels    = [r["label"] for r in sent_recs]
        alphas    = [r["alpha"] for r in sent_recs]

        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1] and labels[i] in ("hi", "en") and labels[i-1] in ("hi", "en"):
                # Switch point at position i
                for off in offsets:
                    j = i + off
                    if 0 <= j < len(alphas):
                        alpha_by_offset[off].append(alphas[j])

    means = [np.mean(alpha_by_offset[o]) if alpha_by_offset[o] else np.nan
             for o in offsets]
    sems  = [np.std(alpha_by_offset[o]) / np.sqrt(len(alpha_by_offset[o]) + 1e-9)
             for o in offsets]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(offsets, means, yerr=sems, marker="o", capsize=4)
    ax.axvline(x=-0.5, color="red", linestyle="--", alpha=0.5, label="switch boundary")
    ax.set_xlabel("Token offset from switch point")
    ax.set_ylabel("Mean α")
    ax.set_title("α trajectory at code-switch boundaries (±3 tokens)")
    ax.legend()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Analysis 3: α vs sentence CMI
# ---------------------------------------------------------------------------

def analysis_cmi_buckets(records: list[dict], task: str, save_path: str, n_buckets: int = 5):
    """Plot α distribution bucketed by sentence CMI."""
    sentences   = _group_by_sentence(records)
    sent_cmis   = {}
    sent_alphas = {}

    for sid, sent_recs in sentences.items():
        sent_recs = sorted(sent_recs, key=lambda r: r["position"])
        sent_labels  = [r["label"] for r in sent_recs]
        sent_alpha_v = [r["alpha"] for r in sent_recs]
        sent_cmis[sid]   = _compute_cmi(sent_labels, task)
        sent_alphas[sid] = sent_alpha_v

    cmis   = np.array(list(sent_cmis.values()))
    if cmis.max() == 0:
        print("  Analysis 3: all CMI=0 (non-LID task or monolingual data), skipping.")
        return

    edges  = np.linspace(0, cmis.max(), n_buckets + 1)
    bucket_labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(n_buckets)]
    buckets = [[] for _ in range(n_buckets)]

    for sid, cmi in sent_cmis.items():
        b = min(int(cmi / (cmis.max() + 1e-9) * n_buckets), n_buckets - 1)
        buckets[b].extend(sent_alphas[sid])

    data   = [b for b in buckets if b]
    labels = [bucket_labels[i] for i, b in enumerate(buckets) if b]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_xlabel("CMI bucket")
    ax.set_ylabel("α")
    ax.set_title("α distribution by sentence CMI")
    plt.xticks(rotation=20)
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Analysis 4: expert disagreement vs α
# ---------------------------------------------------------------------------

def analysis_expert_disagreement(records: list[dict], save_path: str, max_pts: int = 5000):
    """Scatter plot / correlation of L2(h_hing - h_rob) vs α."""
    l2s    = [r["l2_disagreement"] for r in records if r["l2_disagreement"] is not None]
    alphas = [r["alpha"]           for r in records if r["l2_disagreement"] is not None]

    if not l2s:
        print("  Analysis 4: no L2 disagreement data available, skipping.")
        return

    # Subsample for readability
    if len(l2s) > max_pts:
        idx    = np.random.choice(len(l2s), max_pts, replace=False)
        l2s    = [l2s[i]    for i in idx]
        alphas = [alphas[i] for i in idx]

    corr = np.corrcoef(l2s, alphas)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(l2s, alphas, alpha=0.2, s=8, rasterized=True)
    ax.set_xlabel("L2 distance ‖h_hing − h_rob‖")
    ax.set_ylabel("α")
    ax.set_title(f"Expert disagreement vs α  (r = {corr:.3f})")
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Analysis 5: cross-task α comparison
# ---------------------------------------------------------------------------

def analysis_cross_task(records_per_task: dict, save_path: str):
    """Violin plot comparing α distributions across tasks.

    records_per_task: {task_name: list[dict]}
    """
    task_names = list(records_per_task.keys())
    data       = [
        [r["alpha"] for r in records_per_task[t]]
        for t in task_names
    ]
    if not any(data):
        print("  Analysis 5: no data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.violinplot(data, positions=range(len(task_names)), showmedians=True)
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels([t.upper() for t in task_names])
    ax.set_ylabel("α")
    ax.set_title("α distribution across tasks")
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_analyses(model, dataloader, task: str, device, figures_dir: str, exp_id: str = ""):
    """Run all applicable analyses for one (model, task) and save figures."""
    os.makedirs(figures_dir, exist_ok=True)
    prefix = os.path.join(figures_dir, f"{exp_id}_{task}_" if exp_id else f"{task}_")

    print(f"\nExtracting alphas for task={task} ...")
    records = extract_alphas(model, dataloader, device, task)
    if not records:
        print("  No records with alpha — nothing to analyse.")
        return records

    print("  Running analysis 1: α by language ...")
    analysis_alpha_by_language(records, prefix + "alpha_by_lang.png")

    print("  Running analysis 2: switch-point trajectory ...")
    analysis_switch_points(records, prefix + "switch_points.png")

    print("  Running analysis 3: α vs CMI ...")
    analysis_cmi_buckets(records, task, prefix + "cmi_buckets.png")

    print("  Running analysis 4: expert disagreement ...")
    analysis_expert_disagreement(records, prefix + "disagreement.png")

    return records   # caller collects for cross-task analysis (analysis 5)


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------

def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {path}")
