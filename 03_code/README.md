# 03_code — Source Code

All source code lives directly in this folder.
Results (checkpoints, metrics, figures, logs) are written to `../05_results/`.

## Files

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
| `tests/` | Unit tests for word-alignment logic |
| `website/` | Frontend HTML (served by api.py) |
| `scripts/` | Shell wrappers: train / eval / sweep / demo |

## Environment setup

```bash
# From repo root
python -m venv .venv
source .venv/bin/activate
pip install -r 03_code/requirements.txt
```

Hardware: NVIDIA GPU (CUDA) recommended; CPU fallback is automatic. Python 3.10+.

## Quick commands (run from repo root)

```bash
# Train
python 03_code/main.py --task lid --mode train --exp_id R1 --seed 42

# Evaluate
python 03_code/main.py --mode eval --checkpoint 05_results/checkpoints/R1-lid-s42.pt

# Full sweep
python 03_code/main.py --mode sweep

# API server + web demo
python 03_code/api.py
```

See the root `README.md` for full documentation.
