# DL\_MoE — Frozen-Expert Mixture of Experts for Code-Mixed NLP

> A token-level, soft **Mixture-of-Experts (MoE)** framework that blends two frozen pre-trained language models — **HingBERT** (Hindi-English specialist) and **RoBERTa-base** (English generalist) — via a learned router to tackle Hindi-English code-mixed sequence labelling tasks.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tasks & Datasets](#tasks--datasets)
4. [Experiments & Baselines](#experiments--baselines)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Configuration Reference](#configuration-reference)
9. [Outputs](#outputs)
10. [Interpretability Analyses](#interpretability-analyses)
11. [Tests](#tests)

---

## Overview

Hindi-English code-mixing is the practice of alternating between Hindi and English within a single utterance — a common phenomenon in informal Indian social media text. Standard NLP models trained purely on English or Hindi struggle with this mixed register.

This project investigates whether a **frozen dual-expert MoE** — where a language-specific expert (HingBERT) and a general-purpose expert (RoBERTa) are kept fixed, and only a lightweight router MLP is trained — can outperform single-expert baselines on three code-mixed sequence labelling benchmarks.

**Key ideas:**
- Both transformer encoders are **completely frozen**; their representations cost no gradient computation.
- A small **RouterMLP** (two linear layers + GELU) produces a scalar blending weight α ∈ (0, 1) per token.
- The blended representation is `α · h_HingBERT + (1−α) · h_RoBERTa`, passed to a linear task head.
- Only the router and task head (~200K parameters) are trained.

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
                                        RouterMLP
                                    (Linear→GELU→Linear→Sigmoid/τ)
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

### RouterMLP

```
Input: (B, T, D_router)   where D_router = 768 or 1536
  → Linear(D_router, 256)
  → GELU
  → Linear(256, 1)
  → Sigmoid(· / τ)         ← temperature τ controls sharpness
```

- **Soft routing** (default): α is a continuous value; the whole model is differentiable.
- **Hard routing** (R7): Straight-through estimator — hard binary gate in the forward pass, soft gradient in the backward pass.
- **Sentence-level routing** (R6): α is computed once from the concatenated CLS tokens of both experts and broadcast to all token positions.

---

## Tasks & Datasets

| Task | Metric | Dataset | Source |
|------|--------|---------|--------|
| **LID** — Language Identification | Weighted F1 | COMI-LINGUA (HuggingFace `LingoIITGN/COMI-LINGUA`) | per-token `hi` / `en` / `ot` labels |
| **POS** — Part-of-Speech Tagging | Accuracy | Twitter Hindi-English POS (GitHub TSV) | 14-class universal tagset |
| **NER** — Named Entity Recognition | Entity-level F1 (seqeval) | SilentFlame Hindi-English NER (GitHub CSV) | BIO: PERSON, ORG, LOC |

All datasets are **downloaded automatically** at runtime. Devanagari-script sentences are filtered out of COMI-LINGUA to keep the data purely romanized/English code-mixed.

**Train / Val / Test splits:**
- COMI-LINGUA: uses dataset's native train/test split; 10% of training data is carved out as a validation set (seed-controlled).
- POS / NER: 80/20 deterministic train/test split (seed=0); validation carved from training.

---

## Experiments & Baselines

All experiments and their configurations are registered in `configs.py`:

### Baselines

| ID | Model | Frozen? | Description |
|----|-------|---------|-------------|
| **B1** | HingBERT | ❌ No | Full fine-tune — upper bound for single HingBERT |
| **B2** | HingBERT | ✅ Yes | Frozen HingBERT + linear head |
| **B3** | RoBERTa | ✅ Yes | Frozen RoBERTa + linear head |
| **B4** | Fixed 50/50 avg | ✅ Yes | Equal blend of both experts (no router) |

### Router Variants (MoE)

| ID | τ | Router Input | Routing Mode | Description |
|----|----|-------------|--------------|-------------|
| **R1** | 1.0 | both | soft, token-level | Main MoE model |
| **R2** | 0.1 | both | soft, token-level | Very sharp / near-hard |
| **R3** | 0.3 | both | soft, token-level | Sharper routing |
| **R4** | 0.5 | both | soft, token-level | Intermediate |
| **R5** | 2.0 | both | soft, token-level | Softer / near-uniform |
| **R6** | 1.0 | both | soft, **sentence-level** | One α per sentence |
| **R7** | 1.0 | both | **hard**, token-level | Straight-through estimator |

### Ablations (Router Input)

| ID | τ | Router Input | Description |
|----|---|-------------|-------------|
| **A2** | 1.0 | `hing` only | Router sees only HingBERT hidden states |
| **A3** | 1.0 | `rob` only | Router sees only RoBERTa hidden states |

> **A1 == R1** — the full-input MoE is the ablation baseline.

---

## Project Structure

```
DL_MoE/
├── configs.py        # All hyperparameters, task labels, experiment registry
├── data.py           # Dataset loaders, tokenization, alignment, DataLoaders
├── models.py         # FrozenExpert, RouterMLP, MoEModel, baselines, build_model()
├── training.py       # Trainer class (training loop, early stopping, checkpointing)
│                     # + evaluate() function
├── analysis.py       # 5 router interpretability analyses + figure generation
├── main.py           # CLI entry point (train / eval / analysis / sweep modes)
├── requirements.txt  # Python dependencies
├── tests/
│   └── test_alignment.py   # 10 unit tests for word-alignment logic
└── results/
    ├── checkpoints/  # Saved model weights  (<exp_id>-<task>-s<seed>.pt)
    ├── metrics/      # JSON metric files + aggregated.json
    ├── logs/         # Per-epoch CSV training logs
    └── figures/      # Interpretability plots (PNG)
```

### Module Responsibilities

| File | Responsibility |
|------|---------------|
| `configs.py` | Single source of truth for all constants and experiment definitions |
| `data.py` | Downloads / loads datasets, dual-tokenizes with HingBERT + RoBERTa, pads/collates batches |
| `models.py` | Defines `FrozenExpert`, `RouterMLP`, `MoEModel`, single-expert and fixed-avg baselines; `build_model()` factory |
| `training.py` | `Trainer`: AdamW + linear warmup schedule, gradient clipping, early stopping, best-weight restore, CSV logging, checkpoint save/load |
| `analysis.py` | Extracts per-word α values from a trained MoE; produces 5 interpretability figures |
| `main.py` | Argument parsing, orchestrates train / eval / analysis / sweep workflows |

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
pip install -r requirements.txt
```

> **Note:** The first run will download HingBERT (~440 MB) and RoBERTa-base (~500 MB) from HuggingFace Hub. Subsequent runs use the local cache.

**CUDA:** PyTorch will automatically use a GPU if available (`torch.cuda.is_available()`). No special configuration required.

---

## Usage

### 1. Training a single experiment

```bash
# Train the main MoE model (R1) on Language ID task, seed 42
python main.py --task lid --mode train --exp_id R1 --seed 42

# Train with explicit MoE flags (equivalent to R1 above)
python main.py --task lid --mode train --model_mode moe --tau 1.0 --seed 42

# Train a baseline: frozen HingBERT
python main.py --task pos --mode train --exp_id B2 --seed 42

# Quick sanity check (3 epochs only)
python main.py --task lid --mode train --exp_id R1 --seed 42 --max_epochs 3
```

### 2. Evaluating a saved checkpoint

```bash
python main.py --mode eval --checkpoint results/checkpoints/R1-lid-s42.pt
```

### 3. Interpretability analysis

```bash
python main.py --mode analysis --checkpoint results/checkpoints/R1-lid-s42.pt --exp_id R1
```

Figures are saved to `results/figures/`.

### 4. Full experiment sweep

```bash
# Run all experiments × all tasks × all seeds
python main.py --mode sweep

# Run a subset of experiments
python main.py --mode sweep --exp_ids R1 B2 B3
```

Results are aggregated in `results/metrics/aggregated.json`.

### CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `train` / `eval` / `analysis` / `sweep` |
| `--task` | — | `lid` / `ner` / `pos` (required for train/eval) |
| `--seed` | `42` | Random seed for reproducibility |
| `--exp_id` | — | Experiment ID from registry (e.g. `R1`, `B2`) |
| `--model_mode` | `moe` | `moe` / `hingbert` / `roberta` / `fixed_avg` |
| `--tau` | `1.0` | Router temperature |
| `--router_input` | `both` | Router input: `both` / `hing` / `rob` |
| `--sentence_level` | `False` | Use sentence-level (CLS) routing |
| `--hard_routing` | `False` | Use hard binary routing (straight-through) |
| `--no_frozen` | `False` | Unfreeze HingBERT expert (B1 baseline only) |
| `--checkpoint` | — | Path to checkpoint `.pt` file |
| `--batch_size` | `32` | DataLoader batch size |
| `--max_epochs` | `20` | Max training epochs (overrides config) |
| `--exp_ids` | — | Subset of experiment IDs for sweep mode |

---

## Configuration Reference

All global settings live in `configs.py`:

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

After training, results are organized as follows:

```
results/
├── checkpoints/
│   └── R1-lid-s42.pt          # Trainable weights + metadata
├── metrics/
│   ├── R1-lid-s42.json        # {"val": {...}, "test": {...}, "config": {...}}
│   └── aggregated.json        # mean ± std across seeds for each exp/task
├── logs/
│   └── R1-lid-s42.csv         # epoch, train_loss, val_metric
└── figures/
    ├── R1_lid_alpha_by_lang.png
    ├── R1_lid_switch_points.png
    ├── R1_lid_cmi_buckets.png
    └── R1_lid_disagreement.png
```

Checkpoint files store **only the trainable parameters** (router + task head), keeping file sizes small (~1 MB). Expert weights are always re-loaded from HuggingFace on demand.

---

## Interpretability Analyses

Running `--mode analysis` produces four figures that probe *what the router has learned*:

| Analysis | Figure | Description |
|----------|--------|-------------|
| **1 — α by token language** | `alpha_by_lang.png` | KDE plots of α for Hindi / English / Other tokens. Tests if the router assigns higher α (→ HingBERT) to Hindi tokens. |
| **2 — Switch-point trajectory** | `switch_points.png` | Mean α in a ±3 token window around language switch boundaries. Tests if α transitions smoothly across switches. |
| **3 — α vs CMI buckets** | `cmi_buckets.png` | Box plots of α grouped by sentence-level Code-Mixing Index (CMI). Tests if more mixed sentences lead to more balanced routing. |
| **4 — Expert disagreement vs α** | `disagreement.png` | Scatter of ‖h\_hing − h\_rob‖₂ vs α with Pearson correlation. Tests if the router defers to one expert when they disagree more. |

A fifth analysis (`analysis_cross_task`) compares α distributions across tasks via violin plots when multiple task records are available.

---

## Tests

Unit tests verify the correctness of the word-alignment logic before running any experiments:

```bash
python -m pytest tests/ -v
```

The test suite (10 tests in `tests/test_alignment.py`) covers:

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
| `scikit-learn` | ≥ 1.3.0 | F1 / accuracy metrics |
| `seqeval` | ≥ 1.2.2 | Entity-level NER F1 |
| `matplotlib` | ≥ 3.7.0 | Figure generation |
| `seaborn` | ≥ 0.12.0 | KDE plots and styling |
| `pytest` | ≥ 7.4.0 | Unit testing |
