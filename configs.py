HINGBERT = "models_and_data/hing-bert"
ROBERTA  = "models_and_data/roberta-base"

TASK_LABELS = {
    "lid": ["hi", "en", "ot"],
    "ner": [
        "O",
        "B-PERSON", "I-PERSON",
        "B-ORGANIZATION", "I-ORGANIZATION",
        "B-LOCATION", "I-LOCATION",
    ],
    "pos": ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "ADP", "PRON",
            "DET", "CONJ", "PART", "PRON_WH", "PART_NEG", "NUM", "X"],
}

TASK_METRICS = {
    "lid": "weighted_f1",
    "ner": "entity_f1",
    "pos": "accuracy",
}

# Training hyperparams
LR = 2e-4
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 3
MAX_SEQ_LEN = 128
WARMUP_FRACTION = 0.1
GRAD_CLIP = 1.0

SEEDS = [42, 123]
TASKS = ["lid", "pos"]   # NER disabled — add "ner" to re-enable

# Experiment registry: maps exp_id → model config dict
EXPERIMENTS = {
    # Baselines
    "B1": {"model_mode": "hingbert", "frozen": False},
    "B2": {"model_mode": "hingbert", "frozen": True},
    "B3": {"model_mode": "roberta",  "frozen": True},
    "B4": {"model_mode": "fixed_avg"},
    # Router variants
    "R1": {"model_mode": "moe", "tau": 1.0},
    "R2": {"model_mode": "moe", "tau": 0.1},
    "R3": {"model_mode": "moe", "tau": 0.3},
    "R4": {"model_mode": "moe", "tau": 0.5},
    "R5": {"model_mode": "moe", "tau": 2.0},
    "R6": {"model_mode": "moe", "tau": 1.0, "sentence_level": True},
    "R7": {"model_mode": "moe", "tau": 1.0, "hard_routing": True},
    "R8": {"model_mode": "moe", "tau": 1.0, "router_type": "gru"},
    "R9": {"model_mode": "moe", "tau": 1.0, "router_type": "cnn"},
    # Ablations (A1 == R1, reuse its results)
    "A2": {"model_mode": "moe", "tau": 1.0, "router_input": "hing"},
    "A3": {"model_mode": "moe", "tau": 1.0, "router_input": "rob"},
}
