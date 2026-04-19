# 03_code — Source Code

The full source code lives in the **repository root** (one level up from this folder). This directory provides the required submission structure; all runnable scripts and configs point back to the root.

## Source files (repo root)

| File | Role |
|------|------|
| `configs.py` | Hyperparameters and experiment registry |
| `data.py` | Dataset loaders and tokenization |
| `models.py` | FrozenExpert, MoEModel, baselines |
| `routers.py` | RouterMLP, RouterGRU, RouterCNN |
| `training.py` | Trainer, evaluate() |
| `analysis.py` | Interpretability analyses + JSON persistence |
| `main.py` | CLI entry point |
| `api.py` | Flask REST API server |
| `setup_data.py` | Offline asset downloader |

See `scripts/` for ready-to-run shell wrappers and the root `README.md` for full documentation.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Hardware used: NVIDIA GPU (CUDA) recommended; CPU fallback is automatic.
Python: 3.10+

## Quick commands

```bash
# Train
bash 03_code/scripts/train.sh

# Evaluate
bash 03_code/scripts/eval.sh

# Full sweep
bash 03_code/scripts/sweep.sh

# API server + web demo
bash 03_code/scripts/demo.sh
```
