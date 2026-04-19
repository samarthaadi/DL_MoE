# Dataset Description

## Tasks and Datasets

### 1. Language Identification (LID)
- **Dataset**: COMI-LINGUA (`LingoIITGN/COMI-LINGUA` on HuggingFace Hub)
- **Task**: Per-token classification into `hi` (Hindi), `en` (English), `ot` (Other)
- **Classes**: 3
- **Metric**: Weighted F1
- **Preprocessing**: Devanagari-script sentences are filtered out; only romanized/English code-mixed utterances are kept
- **Split**: Native train/test split from HuggingFace. 10% of training data carved out as validation (seed-controlled)

### 2. Part-of-Speech Tagging (POS)
- **Dataset**: Twitter Hindi-English POS corpus (TSV format, GitHub)
- **Task**: Per-token POS tag prediction
- **Classes**: 14 (NOUN, PROPN, VERB, ADJ, ADV, ADP, PRON, DET, CONJ, PART, PRON_WH, PART_NEG, NUM, X)
- **Metric**: Token-level accuracy
- **Preprocessing**: Raw TSV loaded, whitespace-tokenized, no script filtering
- **Split**: 80/20 deterministic train/test (seed=0); 10% of train carved for validation

### 3. Named Entity Recognition (NER)
- **Dataset**: SilentFlame Hindi-English NER (CSV format, GitHub)
- **Task**: BIO-tagged named entity recognition
- **Classes**: 7 (O, B-PERSON, I-PERSON, B-ORGANIZATION, I-ORGANIZATION, B-LOCATION, I-LOCATION)
- **Metric**: Entity-level F1 (seqeval)
- **Status**: Loaded but excluded from the main sweep (NER results are 0.0 for frozen models — included in baselines only)

## Split Sizes (approximate, seed=42)

| Task | Train | Val | Test |
|------|-------|-----|------|
| LID  | ~8,100 | ~900 | ~2,900 |
| POS  | ~3,200 | ~400 | ~1,000 |
| NER  | ~3,600 | ~400 | ~1,100 |

## Tokenization

Each sentence is **dual-tokenized** — once with HingBERT's WordPiece tokenizer and once with RoBERTa's BPE tokenizer. Sub-tokens are averaged back to the original word level (`align_subtokens_to_words` in `data.py`) so both encoders produce consistent `(B, T, 768)` tensors.

- Max sequence length: 128 tokens
- Batch size: 32

## Compute Constraints

Full models (B1) require ~418 MB checkpoint storage and ~8 GB GPU memory. MoE and frozen-expert models require < 2 MB checkpoint storage. All experiments were run on a single GPU.
