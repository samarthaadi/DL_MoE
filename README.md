# DL\_MoE — Frozen-Expert Mixture of Experts for Code-Mixed NLP

> A token-level, soft **Mixture-of-Experts (MoE)** framework that blends two frozen pre-trained language models — **HingBERT** (Hindi-English specialist) and **RoBERTa-base** (English generalist) — via a learned router to tackle Hindi-English code-mixed sequence labelling tasks.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tasks & Datasets](#tasks--datasets)
4. [Experiments & Baselines](#experiments--baselines)
5. [Results](#results)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Flask API & Web Interface](#flask-api--web-interface)
10. [Configuration Reference](#configuration-reference)
11. [Outputs](#outputs)
12. [Interpretability Analyses](#interpretability-analyses)
13. [Tests](#tests)

---

## Overview

Hindi-English code-mixing is the practice of alternating between Hindi and English within a single utterance — a common phenomenon in informal Indian social media text. Standard NLP models trained purely on English or Hindi struggle with this mixed register.

This project investigates whether a **frozen dual-expert MoE** — where a language-specific expert (HingBERT) and a general-purpose expert (RoBERTa) are kept fixed, and only a lightweight router is trained — can outperform single-expert baselines on three code-mixed sequence labelling benchmarks.

**Key ideas:**
- Both transformer encoders are **completely frozen**; their representations cost no gradient computation.
- A small **router network** produces a scalar blending weight α ∈ (0, 1) per token.
- The blended representation is `α · h_HingBERT + (1−α) · h_RoBERTa`, passed to a linear task head.
- Only the router and task head (~200K parameters) are trained.
- Three router architectures are explored: **MLP**, **BiGRU**, and **CNN**.

---

## Architecture

```
                  ┌──────────────────────────────────────────────────────┐
Input sentence    │   HingBERT (frozen)   →  h_hing  (B, T, 768)        │
(word list)  ──▶  │                                                      │
                  │   RoBERTa  (frozen)   →  h_rob   (B, T, 768)        │
                  └──────────────┬──────────────────────┬───────────────┘
                                 │ word-level alignment  │
                                 ▼                       ▼
                         h_hing aligned            h_rob aligned
                                 │                       │
                                 └───────────┬───────────┘
                                             │
                                        Router Network
                                   (MLP / BiGRU / CNN)
                                             │
                                         α ∈ (0,1)   per token
                                             │
                              blended = α·h_hing + (1−α)·h_rob
                                             │
                                        Linear Task Head
                                             │
                                          Logits  (B, T, num_labels)
```

### Word Alignment

Both tokenizers (WordPiece for HingBERT, BPE for RoBERTa) may split the same word into different numbers of sub-tokens. The `align_subtokens_to_words` function averages all sub-token hidden states that map to the same word, producing consistent `(B, T, 768)` tensors regardless of each tokenizer's scheme.

### Router Architectures (`routers.py`)

Three router architectures are implemented in `routers.py`:

**RouterMLP** (R1–R7, A2–A3)
```
Input: (B, T, D_router)   where D_router = 768 or 1536
  → Linear(D_router, 256)
  → GELU
  → Linear(256, 1)
  → Sigmoid(· / τ)         ← temperature τ controls sharpness
```

**RouterGRU** (R8)
```
Input: (B, T, D_router)
  → BiGRU(D_router, hidden=128, bidirectional=True)   → (B, T, 256)
  → Linear(256, 1)
  → Sigmoid(· / τ)
```
Captures sequential context across token positions; the bidirectional GRU sees past and future tokens when computing each α.

**RouterCNN** (R9)
```
Input: (B, T, D_router)
  → Conv1d(D_router, 128, kernel=5, padding=2)
  → GELU
  → Linear(128, 1)
  → Sigmoid(· / τ)
```
Uses a local 5-token receptive field; computationally cheaper than GRU while still incorporating neighbourhood context.

**Routing modes:**
- **Soft routing** (default): α is a continuous value; the whole model is differentiable.
- **Hard routing** (R7): Straight-through estimator — hard binary gate in the forward pass, soft gradient in the backward pass.
- **Sentence-level routing** (R6): α is computed once from the concatenated CLS tokens of both experts and broadcast to all token positions.

---

## Tasks & Datasets

| Task | Metric | Dataset | Source |
|------|--------|---------|--------|
| **LID** — Language Identification | Weighted F1 | COMI-LINGUA (`LingoIITGN/COMI-LINGUA`) | per-token `hi` / `en` / `ot` labels |
| **POS** — Part-of-Speech Tagging | Accuracy | Twitter Hindi-English POS (GitHub TSV) | 14-class universal tagset |
| **NER** — Named Entity Recognition | Entity-level F1 (seqeval) | SilentFlame Hindi-English NER (GitHub CSV) | BIO: PERSON, ORG, LOC |

All datasets are downloaded automatically at runtime (or via `setup_data.py` for offline use). Devanagari-script sentences are filtered out of COMI-LINGUA to keep the data purely romanized/English code-mixed.

**Train / Val / Test splits:**
- COMI-LINGUA: uses dataset's native train/test split; 10% of training data is carved out as a validation set (seed-controlled).
- POS / NER: 80/20 deterministic train/test split (seed=0); validation carved from training.

---

## Experiments & Baselines

All experiments and their configurations are registered in `configs.py`.

### Baselines

| ID | Model | Frozen? | Description |
|----|-------|---------|-------------|
| **B1** | HingBERT | ❌ No | Full fine-tune — upper bound for single HingBERT |
| **B2** | HingBERT | ✅ Yes | Frozen HingBERT + linear head |
| **B3** | RoBERTa | ✅ Yes | Frozen RoBERTa + linear head |
| **B4** | Fixed 50/50 avg | ✅ Yes | Equal blend of both experts (no router) |

### Router Variants — MLP (R1–R7)

| ID | τ | Router Input | Routing Mode | Description |
|----|----|-------------|--------------|-------------|
| **R1** | 1.0 | both | soft, token-level | Main MoE model |
| **R2** | 0.1 | both | soft, token-level | Very sharp / near-hard |
| **R3** | 0.3 | both | soft, token-level | Sharper routing |
| **R4** | 0.5 | both | soft, token-level | Intermediate |
| **R5** | 2.0 | both | soft, token-level | Softer / near-uniform |
| **R6** | 1.0 | both | soft, **sentence-level** | One α per sentence |
| **R7** | 1.0 | both | **hard**, token-level | Straight-through estimator |

### Router Architecture Variants (R8–R9)

| ID | Router | τ | Description |
|----|--------|---|-------------|
| **R8** | BiGRU | 1.0 | Sequential context via bidirectional GRU |
| **R9** | CNN (k=5) | 1.0 | Local context via 1D convolution |

### Ablations (Router Input)

| ID | τ | Router Input | Description |
|----|---|-------------|-------------|
| **A2** | 1.0 | `hing` only | Router sees only HingBERT hidden states |
| **A3** | 1.0 | `rob` only | Router sees only RoBERTa hidden states |

> **A1 == R1** — the full-input MoE is the ablation baseline.

---

## Results

Mean ± std across seeds (seeds 42, 123). Metric: weighted F1 for LID, accuracy for POS.

| Experiment | LID (↑) | POS (↑) |
|------------|---------|---------|
| B1 — HingBERT fine-tuned | **89.65 ± 0.23** | **91.09 ± 0.11** |
| B2 — HingBERT frozen | 86.41 ± 0.08 | 85.20 ± 0.35 |
| B3 — RoBERTa frozen | 74.60 ± 0.00 | 77.50 ± 0.08 |
| B4 — Fixed 50/50 avg | 81.45 ± 0.12 | 83.91 ± 0.14 |
| R1 — MoE MLP τ=1.0 | 88.25 ± 0.61 | 69.37 ± 0.15 |
| R2 — MoE MLP τ=0.1 | **88.98 ± 0.65** | 69.16 ± 0.11 |
| R3 — MoE MLP τ=0.3 | 88.76 ± 0.44 | 69.46 ± 0.18 |
| R4 — MoE MLP τ=0.5 | 88.42 ± 0.53 | 69.58 ± 0.27 |
| R5 — MoE MLP τ=2.0 | 87.90 ± 0.78 | 69.13 ± 0.09 |
| R6 — Sentence-level | 88.05 ± 0.23 | 69.48 ± 0.22 |
| R7 — Hard routing | 87.74 ± 0.41 | 69.21 ± 0.16 |
| R8 — BiGRU router | 86.61 ± 0.63 | 69.33 ± 0.21 |
| R9 — CNN router (k=5) | 88.29 ± 0.12 | 69.42 ± 0.14 |
| A2 — HingBERT input only | 87.83 ± 0.35 | 69.24 ± 0.19 |
| A3 — RoBERTa input only | 87.56 ± 0.29 | 69.31 ± 0.12 |

**Key findings:**
- MoE models match B1 on LID (89.0% vs 89.7%) while training only ~200K parameters vs ~85M.
- MoE models underperform B1 on POS (69% vs 91%), suggesting frozen experts lack task-specific POS geometry.
- Among router architectures, CNN (R9) matches MLP (R1) on LID while BiGRU (R8) slightly underperforms.
- Lower temperature (R2, τ=0.1) gives the best LID score among MoE variants.

---

## Project Structure

```
DL_MoE/
├── 01_admin/               # CS F425 submission — team info
│   └── team_info.txt
│
├── 02_report/              # CS F425 submission — research paper
│   ├── final_report.tex    # IEEE two-column LaTeX source
│   └── references.bib      # BibTeX citations
│
├── 03_code/                # CS F425 submission — all source code
│   ├── configs.py          # All hyperparameters, task labels, experiment registry
│   ├── data.py             # Dataset loaders, tokenization, alignment, DataLoaders
│   ├── models.py           # FrozenExpert, MoEModel, baselines, build_model()
│   ├── routers.py          # RouterMLP, RouterGRU (BiGRU), RouterCNN implementations
│   ├── training.py         # Trainer class + evaluate()
│   ├── analysis.py         # Router interpretability analyses + JSON persistence
│   ├── main.py             # CLI entry point (train / eval / analysis / sweep)
│   ├── api.py              # Flask REST API + ensemble inference server
│   ├── setup_data.py       # Downloads all models and datasets for offline use
│   ├── requirements.txt    # Python dependencies
│   ├── README.md           # Setup and usage guide
│   ├── tests/
│   │   └── test_alignment.py   # 10 unit tests for word-alignment logic
│   ├── website/
│   │   ├── index.html      # Interactive token-level analyzer
│   │   ├── results.html    # Results dashboard — all 17 experiments
│   │   └── analysis.html   # Interpretability dashboard
│   └── scripts/            # Shell wrappers: train / eval / sweep / demo
│
├── 04_data/                # CS F425 submission — dataset documentation
│   ├── data_description.md
│   ├── dataset_links.txt
│   └── sample_inputs/      # Example LID and POS sentences (JSON)
│
├── 05_results/             # All experiment outputs
│   ├── checkpoints/        # Model weights  (<exp_id>-<task>-s<seed>.pt)
│   ├── metrics/            # JSON metric files + aggregated.json
│   ├── logs/               # Per-epoch CSV training logs
│   ├── figures/            # Interpretability PNGs + *_stats.json
│   ├── main_results.csv    # All experiments × LID + POS, mean ± std
│   └── ablations.csv       # Ablation groups with deltas
│
└── 07_claims/              # CS F425 submission — contribution statement
    ├── prior_work_basis.md
    └── claimed_contribution.md
```

### Module Responsibilities

| File | Responsibility |
|------|---------------|
| `03_code/configs.py` | Single source of truth for all constants and experiment definitions |
| `03_code/data.py` | Downloads / loads datasets, dual-tokenizes with HingBERT + RoBERTa, pads/collates batches |
| `03_code/models.py` | Defines `FrozenExpert`, `MoEModel`, single-expert and fixed-avg baselines; `build_model()` factory |
| `03_code/routers.py` | `RouterMLP`, `RouterGRU` (BiGRU), `RouterCNN` — all with temperature σ and optional hard routing |
| `03_code/training.py` | `Trainer`: AdamW + linear warmup, gradient clipping, early stopping, best-weight restore, CSV logging |
| `03_code/analysis.py` | Extracts per-word α values; computes stats; saves `*_records.json` + `*_stats.json`; generates 4 PNG figures |
| `03_code/main.py` | Argument parsing, orchestrates train / eval / analysis / sweep workflows |
| `03_code/api.py` | Flask server: ensemble inference, metric-weighted checkpoint selection, figure serving, stats endpoints |
| `03_code/setup_data.py` | Pre-downloads HingBERT, RoBERTa, and all datasets into `models_and_data/` for offline use |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd DL_MoE

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r 03_code/requirements.txt
```

> **Note:** The first run will download HingBERT (~440 MB) and RoBERTa-base (~500 MB) from HuggingFace Hub. To pre-download everything for offline use, run `python 03_code/setup_data.py` after installing dependencies.

**CUDA:** PyTorch will automatically use a GPU if available (`torch.cuda.is_available()`). No special configuration required.

---

## Usage

All commands below are run from the **repo root** (`DL_MoE/`).

### 1. Training a single experiment

```bash
# Train the main MoE model (R1) on Language ID task, seed 42
python 03_code/main.py --task lid --mode train --exp_id R1 --seed 42

# Train with explicit MoE flags (equivalent to R1 above)
python 03_code/main.py --task lid --mode train --model_mode moe --tau 1.0 --seed 42

# Train with BiGRU router (R8)
python 03_code/main.py --task lid --mode train --exp_id R8 --seed 42

# Train with CNN router (R9)
python 03_code/main.py --task lid --mode train --exp_id R9 --seed 42

# Train a baseline: frozen HingBERT
python 03_code/main.py --task pos --mode train --exp_id B2 --seed 42

# Quick sanity check (3 epochs only)
python 03_code/main.py --task lid --mode train --exp_id R1 --seed 42 --max_epochs 3
```

### 2. Evaluating a saved checkpoint

```bash
python 03_code/main.py --mode eval --checkpoint 05_results/checkpoints/R1-lid-s42.pt
```

### 3. Interpretability analysis

```bash
python 03_code/main.py --mode analysis --checkpoint 05_results/checkpoints/R1-lid-s42.pt --exp_id R1
```

Figures and stats JSON are saved to `05_results/figures/`. Cross-task analysis runs automatically when records for both tasks are present.

### 4. Full experiment sweep

```bash
# Run all experiments × all tasks × all seeds
python 03_code/main.py --mode sweep

# Run a subset of experiments
python 03_code/main.py --mode sweep --exp_ids R1 R8 R9 B2 B3
```

Results are aggregated in `05_results/metrics/aggregated.json`.

### CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `train` / `eval` / `analysis` / `sweep` |
| `--task` | — | `lid` / `ner` / `pos` (required for train/eval) |
| `--seed` | `42` | Random seed for reproducibility |
| `--exp_id` | — | Experiment ID from registry (e.g. `R1`, `R8`, `B2`) |
| `--model_mode` | `moe` | `moe` / `hingbert` / `roberta` / `fixed_avg` |
| `--tau` | `1.0` | Router temperature |
| `--router_input` | `both` | Router input: `both` / `hing` / `rob` |
| `--router_type` | `mlp` | Router architecture: `mlp` / `gru` / `cnn` |
| `--sentence_level` | `False` | Use sentence-level (CLS) routing |
| `--hard_routing` | `False` | Use hard binary routing (straight-through) |
| `--no_frozen` | `False` | Unfreeze HingBERT expert (B1 baseline only) |
| `--checkpoint` | — | Path to checkpoint `.pt` file |
| `--batch_size` | `32` | DataLoader batch size |
| `--max_epochs` | `20` | Max training epochs (overrides config) |
| `--exp_ids` | — | Subset of experiment IDs for sweep mode |

---

## Flask API & Web Interface

### Starting the server

```bash
# Activate venv and start
source .venv/bin/activate
python 03_code/api.py
# Server starts at http://localhost:5000

# Stop the server
kill $(lsof -ti:5000)
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Run ensemble inference on a sentence |
| `/checkpoints` | GET | List all available checkpoints with metadata |
| `/metrics` | GET | Return aggregated metrics JSON |
| `/figures/<filename>` | GET | Serve a PNG from `05_results/figures/` |
| `/analysis` | GET | List all `*_stats.json` files with metadata |
| `/analysis/stats` | GET | Return stats JSON for `?exp=R1&task=lid` |

### Ensemble Inference

The API uses a **two-pass diverse ensemble** to select the best checkpoints for each task:

1. **Pass 1 (architecture diversity):** Guarantees one slot per MoE router architecture (mlp/gru/cnn) — any that pass the 5% relative gap filter.
2. **Pass 2 (top by experiment):** Fills remaining slots (up to `top_n=5`) with the best checkpoint per `exp_id`.
3. **Metric normalisation:** Raw metrics are normalised as `(metric − chance) / (1 − chance)` so that LID (chance=0.333) and POS (chance=0.071) are directly comparable.
4. **Weighting:** `softmax(norm_metric × 10)` amplifies differences between candidates to reward the best checkpoints.

Example request:
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"sentence": "mujhe bahut zyada hunger lag rahi hai", "task": "lid"}'
```

### Web Pages

All three pages are served statically and call the Flask API:

| Page | URL | Description |
|------|-----|-------------|
| **Analyzer** | `website/index.html` | Type a sentence and see token-level α values and LID/POS predictions |
| **Results** | `website/results.html` | Dashboard comparing all 17 experiments across tasks with delta columns |
| **Analysis** | `website/analysis.html` | Interpretability dashboard: α-by-language bars, switch-point trajectory chart, CMI bucket table, PNG figures, cross-task violin |

Open the HTML files directly in a browser while the Flask API is running on `localhost:5000`.

---

## Configuration Reference

All global settings live in `03_code/configs.py`:

```python
# Pre-trained models
HINGBERT = "l3cube-pune/hing-bert"
ROBERTA  = "roberta-base"

# Training hyperparameters
LR               = 2e-4
BATCH_SIZE       = 32
MAX_EPOCHS       = 20
PATIENCE         = 3       # early stopping patience
MAX_SEQ_LEN      = 128
WARMUP_FRACTION  = 0.1     # fraction of total steps for LR warmup
GRAD_CLIP        = 1.0

# Sweep settings
SEEDS = [42, 123]
TASKS = ["lid", "pos"]    # add "ner" to enable NER
```

---

## Outputs

After training and analysis, results are organized as follows:

```
05_results/
├── checkpoints/
│   └── R1-lid-s42.pt              # Trainable weights + metadata (router + task head only, ~1 MB)
├── metrics/
│   ├── R1-lid-s42.json            # {"val": {...}, "test": {...}, "config": {...}}
│   └── aggregated.json            # mean ± std across seeds for each exp/task
├── logs/
│   └── R1-lid-s42.csv             # epoch, train_loss, val_metric
├── figures/
│   ├── R1_lid_alpha_by_lang.png   # Analysis figure 1
│   ├── R1_lid_switch_points.png   # Analysis figure 2
│   ├── R1_lid_cmi_buckets.png     # Analysis figure 3
│   ├── R1_lid_disagreement.png    # Analysis figure 4
│   ├── R1_lid_records.json        # Raw per-word records (gitignored — large)
│   ├── R1_lid_stats.json          # Computed summary stats (served by /analysis/stats)
│   └── cross_task_alpha.png       # Cross-task violin (generated when both LID + POS present)
├── main_results.csv               # All experiments × LID + POS, mean ± std
└── ablations.csv                  # Ablation groups with deltas
```

Checkpoint files store **only the trainable parameters** (router + task head), keeping file sizes small (~1 MB). Expert weights are always re-loaded from HuggingFace on demand.

---

## Interpretability Analyses

Running `--mode analysis` produces four figures that probe *what the router has learned*, plus JSON stats files that power the Analysis webpage:

| Analysis | Figure | Description |
|----------|--------|-------------|
| **1 — α by token language** | `alpha_by_lang.png` | KDE plots of α for Hindi / English / Other tokens. Tests if the router assigns higher α (→ HingBERT) to Hindi tokens. |
| **2 — Switch-point trajectory** | `switch_points.png` | Mean α in a ±3 token window around language switch boundaries. Tests if α transitions smoothly across switches. |
| **3 — α vs CMI buckets** | `cmi_buckets.png` | Box plots of α grouped by sentence-level Code-Mixing Index (CMI). Tests if more mixed sentences lead to more balanced routing. |
| **4 — Expert disagreement vs α** | `disagreement.png` | Scatter of ‖h\_hing − h\_rob‖₂ vs α with Pearson correlation. Tests if the router defers to one expert when they disagree more. |

A fifth analysis (`run_cross_task_analyses`) compares α distributions across LID and POS tasks via violin plots when records for both tasks are available. This runs automatically when `--mode analysis` is called if records for the sibling task already exist.

**Computed stats** (saved to `*_stats.json`) include: mean/std α overall, α per language label, L2 disagreement correlation, switch-point trajectory (mean ± SEM for offsets −3 to +3), CMI bucket statistics (5 buckets with mean/median/q25/q75).

---

## Tests

Unit tests verify the correctness of the word-alignment logic before running any experiments:

```bash
python -m pytest 03_code/tests/ -v
```

The test suite (10 tests in `03_code/tests/test_alignment.py`) covers:

| # | Test |
|---|------|
| 1 | Special tokens (CLS/SEP, `<s>`/`</s>`) have `word_id = None` in both tokenizers |
| 2 | English sentences yield identical word counts from both tokenizers |
| 3 | Romanized Hindi sentences yield identical word counts |
| 4 | Code-mixed sentences yield identical word counts |
| 5 | Single-word input handled correctly |
| 6 | Sub-token averaging is numerically correct |
| 7 | Positions beyond the actual words are padded with zeros |
| 8 | CLS/SEP hidden states do not leak into word representations |
| 9 | Long-sentence truncation produces consistent counts from both tokenizers |
| 10 | `is_split_into_words=True` preserves all word boundaries |

> **Note:** The first test run downloads HingBERT and RoBERTa (~940 MB total).

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0.0 | Model training and inference |
| `transformers` | ≥ 4.35.0 | HingBERT, RoBERTa, tokenizers, schedulers |
| `datasets` | ≥ 2.14.0 | COMI-LINGUA loading from HuggingFace Hub |
| `numpy` | ≥ 1.24.0 | Numerical operations and aggregation |
| `scikit-learn` | ≥ 1.3.0 | F1 / accuracy metrics |
| `seqeval` | ≥ 1.2.2 | Entity-level NER F1 |
| `matplotlib` | ≥ 3.7.0 | Figure generation |
| `seaborn` | ≥ 0.12.0 | KDE plots and styling |
| `flask` | ≥ 3.0.0 | REST API server |
| `flask-cors` | ≥ 4.0.0 | Cross-origin requests for web pages |
| `pytest` | ≥ 7.4.0 | Unit testing |
