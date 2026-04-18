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
"""

import json
import os
import sys
import glob

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Ensure project root is in path
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

_tokenizers = {}      # {"hing": tok, "rob": tok}
_model_cache = {}     # {(task, checkpoint_path): model}


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
    if key not in _model_cache:
        print(f"[API] Building model  task={task}  ckpt={checkpoint_path}")
        exp_config = {"model_mode": "moe", "tau": 1.0}
        model = build_model(exp_config, task, DEVICE)

        if checkpoint_path and os.path.exists(checkpoint_path):
            metrics, _ = Trainer.load_checkpoint(checkpoint_path, model)
            print(f"[API] Checkpoint loaded — saved metrics: {metrics}")
        else:
            print("[API] No checkpoint — using frozen experts + random router.")

        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def _find_best_checkpoint(task: str) -> str | None:
    """Return the best available checkpoint for a task (prefers R1, then any)."""
    ckpt_dir = os.path.join(os.path.dirname(__file__), "results", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    # Prefer R1 first, then any matching checkpoint
    for prefix in (f"R1-{task}-", f"custom-{task}-"):
        matches = glob.glob(os.path.join(ckpt_dir, f"{prefix}*.pt"))
        if matches:
            return matches[0]

    matches = glob.glob(os.path.join(ckpt_dir, f"*-{task}-*.pt"))
    return matches[0] if matches else None


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
    return jsonify({"checkpoints": [os.path.basename(f) for f in files]})


@app.route("/metrics")
def get_metrics():
    metrics_path = os.path.join(
        os.path.dirname(__file__), "results", "metrics", "aggregated.json"
    )
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
        return jsonify(data)

    # Fall back: read individual metrics files
    data = {}
    metrics_dir = os.path.join(os.path.dirname(__file__), "results", "metrics")
    for mf in glob.glob(os.path.join(metrics_dir, "*.json")):
        name = os.path.splitext(os.path.basename(mf))[0]
        with open(mf) as f:
            data[name] = json.load(f)
    return jsonify(data)


@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)
    text       = (body.get("text") or "").strip()
    task       = body.get("task", "lid")
    custom_ckpt = body.get("checkpoint")   # optional explicit path

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if task not in configs.TASK_LABELS:
        return jsonify({"error": f"Unknown task '{task}'"}), 400

    # --- Tokenise --------------------------------------------------------
    words = text.split()
    if not words:
        return jsonify({"error": "No words found after splitting"}), 400

    hing_tok, rob_tok = _get_tokenizers()

    enc_kwargs = dict(
        is_split_into_words=True,
        truncation=True,
        max_length=configs.MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    hing_enc = hing_tok(words, **enc_kwargs)
    rob_enc  = rob_tok(words,  **enc_kwargs)

    hing_wids = hing_enc.word_ids(0)
    rob_wids  = rob_enc.word_ids(0)

    hing_max  = max((w for w in hing_wids if w is not None), default=-1) + 1
    rob_max   = max((w for w in rob_wids  if w is not None), default=-1) + 1
    num_words = min(hing_max, rob_max, len(words))

    if num_words == 0:
        return jsonify({"error": "Tokenization produced 0 words"}), 400

    label_list = configs.TASK_LABELS[task]
    label2id   = {l: i for i, l in enumerate(label_list)}

    batch = {
        "hing_input_ids":      hing_enc.input_ids.to(DEVICE),
        "hing_attention_mask": hing_enc.attention_mask.to(DEVICE),
        "hing_word_ids":       [hing_wids],
        "rob_input_ids":       rob_enc.input_ids.to(DEVICE),
        "rob_attention_mask":  rob_enc.attention_mask.to(DEVICE),
        "rob_word_ids":        [rob_wids],
        # Dummy labels (all 0) — only needed for shape; not used in forward
        "labels":    torch.zeros(1, num_words, dtype=torch.long).to(DEVICE),
        "num_words": torch.tensor([num_words], dtype=torch.long).to(DEVICE),
        "words":     [words[:num_words]],
    }

    # --- Choose checkpoint -----------------------------------------------
    if custom_ckpt:
        ckpt_path = custom_ckpt if os.path.isabs(custom_ckpt) else \
                    os.path.join(os.path.dirname(__file__), "results", "checkpoints", custom_ckpt)
    else:
        ckpt_path = _find_best_checkpoint(task)

    # --- Run model -------------------------------------------------------
    model = _get_model(task, ckpt_path)

    with torch.no_grad():
        out = model(batch)

    logits  = out["logits"][0, :num_words]           # (T, num_labels)
    probs   = torch.softmax(logits, dim=-1)
    preds   = logits.argmax(-1)
    alpha   = out.get("alpha")
    l2_dist = out.get("l2_disagreement")

    # Also capture raw hidden-state expert outputs for display
    # Re-run experts separately (still no_grad — frozen)
    with torch.no_grad():
        h_hing_sub = model.hing_expert(
            batch["hing_input_ids"], batch["hing_attention_mask"]
        )
        h_rob_sub = model.rob_expert(
            batch["rob_input_ids"], batch["rob_attention_mask"]
        )

    h_hing = align_subtokens_to_words(h_hing_sub[0], hing_wids, num_words).cpu()
    h_rob  = align_subtokens_to_words(h_rob_sub[0],  rob_wids,  num_words).cpu()

    # Per-token L2 disagreement (recompute if not provided by model)
    if l2_dist is None:
        l2_vals = torch.norm(h_hing - h_rob, dim=-1).tolist()
    else:
        l2_vals = l2_dist[0, :num_words].cpu().tolist()

    # --- Build response --------------------------------------------------
    tokens = []
    for i in range(num_words):
        pred_id    = preds[i].item()
        pred_label = label_list[pred_id]
        confidence = probs[i, pred_id].item()

        token_alpha = None
        if alpha is not None:
            token_alpha = (
                alpha[0, i].item() if alpha.dim() == 2 else alpha[0].item()
            )

        tokens.append({
            "word":            words[i],
            "prediction":      pred_label,
            "confidence":      round(confidence, 4),
            "alpha":           round(token_alpha, 4) if token_alpha is not None else None,
            "l2_disagreement": round(l2_vals[i], 4),
            "all_probs":       {
                label_list[j]: round(probs[i, j].item(), 4)
                for j in range(len(label_list))
            },
        })

    return jsonify({
        "tokens":         tokens,
        "task":           task,
        "num_words":      num_words,
        "checkpoint":     os.path.basename(ckpt_path) if ckpt_path else None,
        "has_checkpoint": ckpt_path is not None and os.path.exists(ckpt_path) if ckpt_path else False,
        "device":         DEVICE,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "website"), exist_ok=True)
    print(f"[API] Starting server on http://localhost:5000  (device={DEVICE})")
    app.run(host="0.0.0.0", port=5000, debug=False)
