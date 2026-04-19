# Prior Work Basis

## Core Papers

### 1. Mixture of Experts (foundational)
**Shazeer et al. (2017)** — *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*
- Introduced the sparse MoE layer with per-example routing via a gating network.
- Influenced our design of per-token routing weight α that blends two expert outputs.
- Our router is simpler (single soft scalar, not K-of-N sparse), adapted for lightweight inference.

### 2. HingBERT — Code-Mixed Pre-training
**Lal et al. (2021)** — *HingBERT, HingRoBERTa, HingRoBERTa-Mixed, MuRIL: Language Models for Hindi-English Code-Mixed NLP*
- Introduced HingBERT, trained on Hindi-English code-mixed social media text.
- We use HingBERT as the "Hindi-specialist" frozen expert.
- Code-mixing phenomena described here (CMI, language switch patterns) directly motivated our interpretability analyses.

### 3. RoBERTa — Robustly Optimized BERT
**Liu et al. (2019)** — *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- RoBERTa-base serves as the English-generalist frozen expert in our system.
- We freeze it to avoid catastrophic forgetting of English representations.

### 4. BERT — Bidirectional Encoder
**Devlin et al. (2018)** — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- Foundational architecture underlying both HingBERT and RoBERTa.
- Word-alignment post-processing (WordPiece sub-token averaging) is motivated by BERT's tokenization scheme.

### 5. Code-Mixed NLP Benchmarks
**Aguilar et al. (2020)** — *LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation*
- Framed evaluation methodology for code-mixed sequence labelling (LID, NER, POS).
- Our task selection and metric choices (weighted F1 for LID, accuracy for POS, seqeval for NER) follow this benchmark's conventions.

### 6. Straight-Through Estimator (for R7)
**Bengio et al. (2013)** — *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*
- Theoretical basis for hard routing with gradients (R7 experiment).
- We implement the straight-through estimator: hard binary gate forward, soft gradient backward.
