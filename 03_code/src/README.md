# Source Files

All Python source files live in the **repository root** (one level up). This avoids breaking relative imports across modules.

| File | Description |
|------|-------------|
| `../../configs.py` | Hyperparameters and experiment registry |
| `../../data.py` | Dataset loaders, tokenization, alignment |
| `../../models.py` | FrozenExpert, MoEModel, baselines, build_model() |
| `../../routers.py` | RouterMLP, RouterGRU, RouterCNN |
| `../../training.py` | Trainer class, evaluate() |
| `../../analysis.py` | Interpretability analyses |
| `../../main.py` | CLI entry point |
| `../../api.py` | Flask REST API |
| `../../setup_data.py` | Offline asset downloader |
| `../../tests/test_alignment.py` | Unit tests |
