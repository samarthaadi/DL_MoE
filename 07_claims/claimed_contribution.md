# Claimed Contributions

## What We Reproduced

- The standard BERT fine-tuning setup (B1) for Hindi-English code-mixed LID and POS tagging, using HingBERT as the base model.
- Evaluation metrics: weighted F1 for LID, token accuracy for POS, seqeval entity F1 for NER — following LinCE conventions.
- Word-level sub-token alignment (averaging WordPiece/BPE hidden states per original word) as described in standard BERT fine-tuning literature.

## What We Modified or Extended

- **Frozen dual-expert MoE**: We adapted the MoE concept from Shazeer et al. to a two-expert, single-scalar-gate setting where both experts are kept completely frozen. Only the router (~200K parameters) is trained, making training ~400× cheaper in terms of trainable parameters than a full B1 fine-tune.
- **Three router architectures**: We implemented and compared RouterMLP (the main model), RouterGRU (bidirectional GRU for sequential context), and RouterCNN (1D convolution with k=5 for local context). This architectural comparison is not present in prior code-mixed MoE work.
- **Temperature sweep**: Systematic ablation of sigmoid temperature τ ∈ {0.1, 0.3, 0.5, 1.0, 2.0} to characterise the sharpness-performance trade-off.
- **Interpretability analyses**: Five analyses probing what the router learns — α by language label, switch-point trajectories, CMI-bucket routing behaviour, expert disagreement correlation, and cross-task α comparison — implemented as reusable code with JSON persistence.
- **Metric-normalised ensemble**: A two-pass diverse ensemble inference system that selects checkpoints using above-chance-normalised metrics, ensuring architecture diversity (MLP/GRU/CNN) in the ensemble.

## What Did Not Work

- **MoE on POS**: Frozen expert representations appear to lack the task-specific geometric structure needed for POS tagging. MoE models achieve ~67% accuracy versus B1's 91%. Unfreezing the experts (B1) is necessary for POS.
- **NER with frozen experts**: All frozen-expert models (B2, B3, B4, MoE) achieve near-zero entity F1 on NER. The task requires fine-grained entity boundary detection that the frozen representations do not capture without task-specific fine-tuning.
- **Hard routing (R7) on POS**: Hard binary gating hurts POS accuracy (65.2%) relative to soft routing, likely because discrete routing loses gradient signal for the task head.
- **Sentence-level routing (R6)**: Broadcasting a single CLS-based α to all tokens reduces LID performance (81.2%) relative to token-level routing, confirming that per-token routing is important for code-mixing where language alternates at word level.

## What We Believe Is Our Contribution

1. A systematic empirical study of frozen dual-expert MoE routing for Hindi-English code-mixed sequence labelling, covering 15 experimental configurations across two tasks.
2. Demonstration that a 200K-parameter frozen MoE router matches a fully fine-tuned HingBERT (89.0% vs 89.7% LID F1) while training only 0.24% as many parameters.
3. The first (to our knowledge) direct comparison of MLP, BiGRU, and CNN routers in a code-mixed MoE setting, showing that CNN (R9) matches MLP at lower sequential cost.
4. An interpretability toolkit that quantifies router behaviour at language boundaries, across CMI levels, and across tasks — contributing reusable analysis code for future code-mixed MoE research.
