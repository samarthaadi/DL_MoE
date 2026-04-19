"""
Flask API backend for the Hinglish MoE Analyzer website.

Endpoints:
  GET  /              → serve website/index.html
  POST /analyze       → run inference, return per-token analysis
  GET  /checkpoints   → list available checkpoint files
  GET  /metrics       → return aggregated training metrics

Run:
  python api.py
  # or with the project venv:
  source ds_env/bin/activate && python api.py

Ensemble mode (POST /analyze with {"ensemble": true}):
  Selects the best checkpoint per experiment type (diverse) that falls
  within 5% relative of the best normalized metric.

  Weighting logic:
    norm_metric = (test_metric - chance) / (1 - chance)
      where chance = 1 / num_labels  (LID: 0.333, POS: 0.071)
    weight = softmax(norm_metric * 10)

  The 5% gap filter is critical: for POS, MoE models score ~0.69 vs HingBERT's
  0.91, so they are automatically excluded and cannot drag accuracy down.
  For LID all top models are within 1-2%, so a diverse mix contributes.
"""

import json
import os
import sys
import glob

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs
from models import build_model, align_subtokens_to_words
from training import Trainer

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="website", static_url_path="")
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[API] Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Lazy model / tokenizer cache
# ---------------------------------------------------------------------------

_tokenizers = {}       # {"hing": tok, "rob": tok}
_model_cache = {}      # {(task, checkpoint_path): model}


def _get_tokenizers():
    global _tokenizers
    if not _tokenizers:
        from transformers import AutoTokenizer
        print("[API] Loading tokenizers …")
        _tokenizers["hing"] = AutoTokenizer.from_pretrained(configs.HINGBERT)
        _tokenizers["rob"]  = AutoTokenizer.from_pretrained(
            configs.ROBERTA, add_prefix_space=True
        )
        print("[API] Tokenizers ready.")
    return _tokenizers["hing"], _tokenizers["rob"]


def _get_model(task: str, checkpoint_path: str | None):
    key = (task, checkpoint_path)
    if key in _model_cache:
        return _model_cache[key]

    print(f"[API] Building model  task={task}  ckpt={checkpoint_path}")

    if checkpoint_path is None:
        exp_config = {"model_mode": "moe", "tau": 1.0}
        model = build_model(exp_config, task, DEVICE)
        print("[API] No checkpoint — using frozen experts + random router.")
    else:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path!r}. "
                "Refusing to fall back to a default model when a checkpoint was requested."
            )

        ckpt_data  = torch.load(checkpoint_path, map_location="cpu")
        exp_config = ckpt_data.get("exp_config") or {}
        ckpt_task  = ckpt_data.get("task")

        if not exp_config:
            raise ValueError(
                f"Checkpoint {checkpoint_path!r} has no 'exp_config'. "
                "Re-train with current code (Trainer.save_checkpoint persists exp_config)."
            )
        if ckpt_task is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path!r} has no 'task' field. "
                "Cannot verify task-specific label decoding."
            )
        if ckpt_task != task:
            raise ValueError(
                f"Checkpoint task {ckpt_task!r} does not match requested task {task!r}."
            )

        model = build_model(exp_config, task, DEVICE)
        metrics, _ = Trainer.load_checkpoint(checkpoint_path, model)
        print(f"[API] Loaded  config={exp_config}  metrics={metrics}")

    model.eval()
    _model_cache[key] = model
    return model


# ---------------------------------------------------------------------------
# Checkpoint scoring and ensemble candidate selection
# ---------------------------------------------------------------------------

def _chance_level(task: str) -> float:
    """Random-chance baseline: 1 / num_labels."""
    return 1.0 / len(configs.TASK_LABELS[task])


def _norm_metric(metric: float, task: str) -> float:
    """Above-chance normalized metric: (m - chance) / (1 - chance).

    Puts metrics on a task-agnostic scale so LID and POS scores are
    comparable before computing softmax ensemble weights.
    """
    chance = _chance_level(task)
    return (metric - chance) / (1.0 - chance)


def _score_checkpoints(task: str) -> list[tuple[str, float, float, dict]]:
    """Return (checkpoint_path, raw_metric, norm_metric, exp_config) sorted by norm_metric desc.

    Reads results/metrics/*-<task>-*.json, pairs each with its .pt file.
    Falls back to listing .pt files with metric=0.5 if no metrics found.
    """
    ckpt_dir    = os.path.join(os.path.dirname(__file__), "results", "checkpoints")
    metrics_dir = os.path.join(os.path.dirname(__file__), "results", "metrics")

    if not os.path.isdir(ckpt_dir):
        return []

    scored = []
    if os.path.isdir(metrics_dir):
        for mf in glob.glob(os.path.join(metrics_dir, f"*-{task}-*.json")):
            try:
                with open(mf) as f:
                    data = json.load(f)
                raw = float(data.get("test", {}).get("metric", -1.0))
                name = os.path.splitext(os.path.basename(mf))[0]   # e.g. "R1-lid-s42"
                cp   = os.path.join(ckpt_dir, f"{name}.pt")
                cfg  = data.get("config", {})
                if os.path.exists(cp) and raw > 0:
                    scored.append((cp, raw, _norm_metric(raw, task), cfg))
            except Exception:
                continue

    scored.sort(key=lambda x: x[2], reverse=True)   # sort by norm_metric

    if not scored:
        for cp in glob.glob(os.path.join(ckpt_dir, f"*-{task}-*.pt")):
            chance = _chance_level(task)
            scored.append((cp, chance, 0.0, {}))

    return scored


def _find_best_checkpoint(task: str) -> str | None:
    """Return the single checkpoint with the highest normalized metric for the task."""
    scored = _score_checkpoints(task)
    return scored[0][0] if scored else None


def _get_diverse_ensemble_candidates(
    task: str, max_gap: float = 0.05, top_n: int = 5
) -> list[tuple[str, float, float, dict]]:
    """Return up to top_n diverse checkpoints for ensemble inference.

    Selection strategy (two passes):
      Pass 1 — Architecture guarantee:
        For each router architecture (mlp, gru, cnn), pick the single best
        MoE checkpoint that passes the quality threshold.  This ensures the
        BiGRU and CNN routers are represented when they qualify, even if many
        high-scoring MLP experiments would otherwise fill all top_n slots.

      Pass 2 — Fill remaining slots by exp_id diversity:
        Among remaining candidates (not yet selected), add the best-scoring
        checkpoint per experiment ID until top_n is reached.

      Quality threshold:
        norm_metric >= best_norm * (1 - max_gap)
        e.g. LID: threshold ≈ 0.806 → R8/GRU (0.809) included, B2 (0.709) not
             POS: threshold ≈ 0.860 → only B1/HingBERT passes (MoE ~ 0.64)

    Returns list of (checkpoint_path, raw_metric, norm_metric, exp_config).
    """
    all_scored = _score_checkpoints(task)
    if not all_scored:
        return []

    best_norm = all_scored[0][2]
    threshold = best_norm * (1.0 - max_gap)
    candidates = [e for e in all_scored if e[2] >= threshold]

    selected: dict[str, tuple] = {}   # exp_id → entry

    # Pass 1: one slot per MoE router architecture (mlp / gru / cnn)
    arch_seen: set[str] = set()
    for entry in candidates:
        cp, raw, norm, cfg = entry
        if cfg.get("model_mode") != "moe":
            continue
        arch = cfg.get("router_type", "mlp")
        if arch not in arch_seen:
            arch_seen.add(arch)
            exp_id = os.path.basename(cp).split("-")[0]
            selected[exp_id] = entry

    # Pass 2: fill remaining slots with best per exp_id (any model_mode)
    for entry in candidates:
        if len(selected) >= top_n:
            break
        cp, raw, norm, cfg = entry
        exp_id = os.path.basename(cp).split("-")[0]
        if exp_id not in selected:
            selected[exp_id] = entry

    diverse = sorted(selected.values(), key=lambda x: x[2], reverse=True)
    return diverse[:top_n]


# ---------------------------------------------------------------------------
# Weighted ensemble inference
# ---------------------------------------------------------------------------

def _run_ensemble(
    batch: dict,
    task: str,
    scored: list[tuple[str, float, float, dict]],
    num_words: int,
    hing_wids: list,
    rob_wids: list,
) -> dict:
    """Run inference through each checkpoint; return norm-metric-weighted outputs.

    Weighting:
      weight_i = softmax(norm_metric_i * 10)
      norm_metric = (raw_metric - chance) / (1 - chance)

    Using norm_metric (above-chance) instead of raw metric makes weights
    task-agnostic and correctly amplifies differences between candidates.

    Returns:
        blended_logits : (T, num_labels) tensor
        blended_alpha  : (T,) tensor or None
        l2_dist        : list[float] or None
        model_infos    : list of dicts with checkpoint metadata + weight
    """
    logit_entries = []   # (norm_metric, logits (T, num_labels))
    alpha_entries = []   # (norm_metric, alpha  (T,))
    l2_dist       = None
    model_infos   = []

    with torch.no_grad():
        for cp, raw_metric, norm, cfg in scored:
            try:
                model = _get_model(task, cp)
            except (ValueError, FileNotFoundError) as e:
                print(f"[API] Skipping {cp}: {e}")
                continue

            out    = model(batch)
            logits = out["logits"][0, :num_words].cpu()
            alpha  = out.get("alpha")
            raw_l2 = out.get("l2_disagreement")

            logit_entries.append((norm, logits))

            if alpha is not None:
                a = alpha[0, :num_words].cpu() if alpha.dim() == 2 \
                    else alpha[0].expand(num_words).cpu()
                alpha_entries.append((norm, a))

            # Capture L2 disagreement from the best dual-expert model seen so far
            if l2_dist is None:
                if raw_l2 is not None:
                    l2_dist = raw_l2[0, :num_words].cpu().tolist()
                elif hasattr(model, "hing_expert") and hasattr(model, "rob_expert"):
                    h_hing_sub = model.hing_expert(
                        batch["hing_input_ids"], batch["hing_attention_mask"]
                    )
                    h_rob_sub = model.rob_expert(
                        batch["rob_input_ids"], batch["rob_attention_mask"]
                    )
                    h_hing = align_subtokens_to_words(
                        h_hing_sub[0], hing_wids, num_words
                    ).cpu()
                    h_rob = align_subtokens_to_words(
                        h_rob_sub[0], rob_wids, num_words
                    ).cpu()
                    l2_dist = torch.norm(h_hing - h_rob, dim=-1).tolist()

            mode = cfg.get("model_mode", "moe")
            model_infos.append({
                "checkpoint":  os.path.basename(cp),
                "raw_metric":  round(raw_metric, 4),
                "norm_metric": round(norm, 4),
                "router_type": cfg.get("router_type", "mlp") if mode == "moe" else "none",
                "model_mode":  mode,
                "tau":         cfg.get("tau", 1.0) if mode == "moe" else None,
            })

    if not logit_entries:
        return {}

    def _softmax_weights(norms: list[float]) -> list[float]:
        t = torch.tensor(norms, dtype=torch.float32)
        return torch.softmax(t * 10.0, dim=0).tolist()

    logit_weights  = _softmax_weights([n for n, _ in logit_entries])
    stacked_logits = torch.stack([l for _, l in logit_entries])    # (N, T, C)
    w_l = torch.tensor(logit_weights)[:, None, None]
    blended_logits = (w_l * stacked_logits).sum(0)                 # (T, C)

    # Record final weights in model_infos now that we know all candidates
    for i, info in enumerate(model_infos):
        info["weight"] = round(logit_weights[i], 4)

    if alpha_entries:
        alpha_weights  = _softmax_weights([n for n, _ in alpha_entries])
        stacked_alphas = torch.stack([a for _, a in alpha_entries])  # (N, T)
        w_a = torch.tensor(alpha_weights)[:, None]
        blended_alpha  = (w_a * stacked_alphas).sum(0)               # (T,)
    else:
        blended_alpha = None

    return {
        "blended_logits": blended_logits,
        "blended_alpha":  blended_alpha,
        "l2_dist":        l2_dist,
        "model_infos":    model_infos,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("website", "index.html")


@app.route("/checkpoints")
def list_checkpoints():
    ckpt_dir = os.path.join(os.path.dirname(__file__), "results", "checkpoints")
    files = glob.glob(os.path.join(ckpt_dir, "*.pt")) if os.path.isdir(ckpt_dir) else []
    return jsonify({"checkpoints": [os.path.basename(f) for f in sorted(files)]})


@app.route("/metrics")
def get_metrics():
    agg_path = os.path.join(
        os.path.dirname(__file__), "results", "metrics", "aggregated.json"
    )
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            return jsonify(json.load(f))

    data = {}
    metrics_dir = os.path.join(os.path.dirname(__file__), "results", "metrics")
    for mf in glob.glob(os.path.join(metrics_dir, "*.json")):
        name = os.path.splitext(os.path.basename(mf))[0]
        with open(mf) as f:
            data[name] = json.load(f)
    return jsonify(data)


@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Serve a PNG figure from results/figures/."""
    from flask import send_from_directory
    figures_dir = os.path.join(os.path.dirname(__file__), "results", "figures")
    return send_from_directory(figures_dir, filename)


@app.route("/analysis")
def list_analyses():
    """List all available analysis results (exp_id + task combos with stats JSON)."""
    figures_dir = os.path.join(os.path.dirname(__file__), "results", "figures")
    analyses = []
    if os.path.isdir(figures_dir):
        for sf in sorted(glob.glob(os.path.join(figures_dir, "*_stats.json"))):
            try:
                with open(sf) as f:
                    s = json.load(f)
                analyses.append({
                    "exp_id":      s.get("exp_id", "?"),
                    "task":        s.get("task", "?"),
                    "num_records": s.get("num_records", 0),
                    "mean_alpha":  s.get("mean_alpha"),
                    "generated_at": s.get("generated_at"),
                    "stats_file":  os.path.basename(sf),
                    "figures":     s.get("figures", {}),
                })
            except Exception:
                continue

    cross_task_path = os.path.join(figures_dir, "cross_task_stats.json")
    cross_task = None
    if os.path.exists(cross_task_path):
        with open(cross_task_path) as f:
            cross_task = json.load(f)

    return jsonify({"analyses": analyses, "cross_task": cross_task})


@app.route("/analysis/stats")
def get_analysis_stats():
    """Return stats JSON for one (exp_id, task) pair.

    Query params: ?exp=R1&task=lid
    """
    exp_id = request.args.get("exp", "")
    task   = request.args.get("task", "")
    figures_dir = os.path.join(os.path.dirname(__file__), "results", "figures")
    stats_path  = os.path.join(figures_dir, f"{exp_id}_{task}_stats.json")
    if not os.path.exists(stats_path):
        return jsonify({"error": f"No stats found for exp={exp_id} task={task}"}), 404
    with open(stats_path) as f:
        return jsonify(json.load(f))


@app.route("/analyze", methods=["POST"])
def analyze():
    body         = request.get_json(force=True)
    text         = (body.get("text") or "").strip()
    task         = body.get("task", "lid")
    custom_ckpt  = body.get("checkpoint")      # optional explicit path
    use_ensemble = bool(body.get("ensemble", False))

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if task not in configs.TASK_LABELS:
        return jsonify({"error": f"Unknown task '{task}'"}), 400

    words = text.split()
    if not words:
        return jsonify({"error": "No words found after splitting"}), 400

    # --- Tokenise --------------------------------------------------------
    hing_tok, rob_tok = _get_tokenizers()

    enc_kwargs = dict(
        is_split_into_words=True,
        truncation=True,
        max_length=configs.MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    hing_enc  = hing_tok(words, **enc_kwargs)
    rob_enc   = rob_tok(words,  **enc_kwargs)
    hing_wids = hing_enc.word_ids(0)
    rob_wids  = rob_enc.word_ids(0)

    hing_max  = max((w for w in hing_wids if w is not None), default=-1) + 1
    rob_max   = max((w for w in rob_wids  if w is not None), default=-1) + 1
    num_words = min(hing_max, rob_max, len(words))

    if num_words == 0:
        return jsonify({"error": "Tokenization produced 0 words"}), 400

    label_list = configs.TASK_LABELS[task]

    batch = {
        "hing_input_ids":      hing_enc.input_ids.to(DEVICE),
        "hing_attention_mask": hing_enc.attention_mask.to(DEVICE),
        "hing_word_ids":       [hing_wids],
        "rob_input_ids":       rob_enc.input_ids.to(DEVICE),
        "rob_attention_mask":  rob_enc.attention_mask.to(DEVICE),
        "rob_word_ids":        [rob_wids],
        "labels":    torch.zeros(1, num_words, dtype=torch.long).to(DEVICE),
        "num_words": torch.tensor([num_words], dtype=torch.long).to(DEVICE),
        "words":     [words[:num_words]],
    }

    # --- Resolve checkpoint list ----------------------------------------
    if custom_ckpt:
        cp = custom_ckpt if os.path.isabs(custom_ckpt) else \
             os.path.join(os.path.dirname(__file__), "results", "checkpoints", custom_ckpt)
        # Single custom checkpoint: raw=1.0, norm=1.0, no config known yet
        scored = [(cp, 1.0, 1.0, {})]
        use_ensemble = False
    elif use_ensemble:
        # Diverse top-5: best per experiment within 5% of best norm_metric
        scored = _get_diverse_ensemble_candidates(task, max_gap=0.05, top_n=5)
    else:
        scored = _score_checkpoints(task)[:1]   # single best by norm_metric

    if not scored:
        return jsonify({"error": f"No checkpoints found for task '{task}'"}), 404

    # --- Inference -------------------------------------------------------
    result = _run_ensemble(batch, task, scored, num_words, hing_wids, rob_wids)
    if not result:
        return jsonify({"error": "All checkpoints failed to load"}), 500

    blended_logits = result["blended_logits"]  # (T, num_labels)
    blended_alpha  = result["blended_alpha"]   # (T,) or None
    l2_dist        = result["l2_dist"]
    model_infos    = result["model_infos"]

    probs = torch.softmax(blended_logits, dim=-1)
    preds = blended_logits.argmax(-1)

    # --- Build per-token response ----------------------------------------
    tokens = []
    for i in range(num_words):
        pred_id    = preds[i].item()
        pred_label = label_list[pred_id]
        confidence = probs[i, pred_id].item()
        token_alpha = blended_alpha[i].item() if blended_alpha is not None else None

        tokens.append({
            "word":            words[i],
            "prediction":      pred_label,
            "confidence":      round(confidence, 4),
            "alpha":           round(token_alpha, 4) if token_alpha is not None else None,
            "l2_disagreement": round(l2_dist[i], 4) if l2_dist is not None else None,
            "all_probs": {
                label_list[j]: round(probs[i, j].item(), 4)
                for j in range(len(label_list))
            },
        })

    return jsonify({
        "tokens":      tokens,
        "task":        task,
        "num_words":   num_words,
        "checkpoint":  model_infos[0]["checkpoint"] if model_infos else None,
        "has_checkpoint": bool(model_infos),
        "device":      DEVICE,
        "ensemble":    use_ensemble,
        "models_used": model_infos,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "website"), exist_ok=True)
    print(f"[API] Starting server on http://localhost:5000  (device={DEVICE})")
    app.run(host="0.0.0.0", port=5000, debug=False)
