"""
Data loading, tokenization, alignment, and DataLoader construction.

Key design: sentences are pre-split into word lists, then passed to both
tokenizers with is_split_into_words=True. This ensures consistent word
indexing via word_ids() regardless of each tokenizer's sub-word scheme.
"""

import ast
import csv
import random
import re

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset

import configs

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

_NER_TAG_MAP = {
    "Other": "O",
    "B-Per": "B-PERSON",  "I-Per": "I-PERSON",
    "B-Org": "B-ORGANIZATION", "I-Org": "I-ORGANIZATION",
    "B-Loc": "B-LOCATION", "I-Loc": "I-LOCATION",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _parse_annotation(ann):
    """Parse annotation field (str or list) → (words, labels).

    Handles two formats found in COMI-LINGUA:
      - list of dicts:  [{'key': word, 'value': label, ...}, ...]
      - list of lists:  [[word, label], ...]
    """
    if isinstance(ann, list):
        items = ann
    elif isinstance(ann, str):
        items = ast.literal_eval(ann)
    else:
        return None, None
    if not items:
        return [], []
    if isinstance(items[0], dict):
        if "key" in items[0]:          # LID format
            words  = [item["key"]    for item in items]
            labels = [item["value"]  for item in items]
        elif "word" in items[0]:       # NER / POS format
            words  = [item["word"]   for item in items]
            labels = [item["entity"] for item in items]
        else:
            return None, None
    elif isinstance(items[0], (list, tuple)):
        words  = [item[0] for item in items]
        labels = [item[1] for item in items]
    else:
        return None, None
    return words, labels


def _flat_ner_to_bio(labels):
    """Convert flat NER type labels to BIO format.
    Consecutive tokens of the same type → same span (B- then I-).
    'X' → 'O'.
    """
    bio, prev = [], None
    for lab in labels:
        if lab == "X":
            bio.append("O")
            prev = None
        elif lab == prev:
            bio.append(f"I-{lab}")
        else:
            bio.append(f"B-{lab}")
            prev = lab
    return bio


def load_comi_lingua(task: str):
    """Load and filter COMI-LINGUA for one task.

    Returns (train_samples, test_samples) where each sample is
    (words: list[str], labels: list[str]).
    """
    print(f"\nLoading COMI-LINGUA/{task.upper()} ...")
    ds = load_from_disk(f"models_and_data/comi_lingua/{task.upper()}")

    # Detect the annotation column (first one mentioning 'Annotator 1')
    cols = ds["train"].column_names
    ann_col = next((c for c in cols if "Annotator 1" in c), None)
    if ann_col is None:
        raise ValueError(f"No 'Annotator 1' column found. Available: {cols}")
    print(f"  Annotation column: '{ann_col}'")

    results = {}
    for split in ("train", "test"):
        samples, n_total, n_deva, n_bad = [], 0, 0, 0
        for row in ds[split]:
            n_total += 1
            ann = row[ann_col]
            if ann is None:
                n_bad += 1
                continue

            words, labels = _parse_annotation(ann)
            if words is None or len(words) == 0 or len(words) != len(labels):
                n_bad += 1
                continue

            if DEVANAGARI_RE.search(" ".join(words)):
                n_deva += 1
                continue

            # Remap unknown labels to the first label in the set (log once)
            known = set(configs.TASK_LABELS[task.lower()])
            if task.lower() == "ner":
                labels = _flat_ner_to_bio(labels)
                known = set(configs.TASK_LABELS["ner"])

            clean_labels = []
            for lab in labels:
                if lab not in known:
                    clean_labels.append(configs.TASK_LABELS[task.lower()][0])
                else:
                    clean_labels.append(lab)
            labels = clean_labels

            samples.append((words, labels))

        print(f"  {split}: {len(samples)}/{n_total} kept  "
              f"({n_deva} Devanagari, {n_bad} malformed dropped)")
        results[split] = samples

    return results["train"], results["test"]


def load_ner_data():
    """Load SilentFlame Hindi-English NER from local file; return (train_all, test_samples).

    80/20 deterministic split (seed=0). Tags mapped to standard BIO.
    """
    print("\nLoading SilentFlame NER ...")
    sentences: dict = {}
    with open("models_and_data/annotatedData.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid  = (row.get("Sent") or "").strip()
            word = (row.get("Word") or "").strip()
            tag  = _NER_TAG_MAP.get((row.get("Tag") or "").strip(), "O")
            if not sid or not word:
                continue
            if sid not in sentences:
                sentences[sid] = ([], [])
            sentences[sid][0].append(word)
            sentences[sid][1].append(tag)

    samples = [(w, l) for w, l in sentences.values() if w]
    rng = random.Random(0)
    rng.shuffle(samples)
    n_test = max(1, int(len(samples) * 0.2))
    train_all, test_samples = samples[n_test:], samples[:n_test]
    print(f"  {len(train_all)} train_all, {len(test_samples)} test")
    return train_all, test_samples


def load_pos_data():
    """Load Twitter Hindi-English POS from local file; return (train_all, test_samples).

    80/20 deterministic split (seed=0). TSV: word<TAB>lang<TAB>pos.
    """
    print("\nLoading Twitter POS ...")
    with open("models_and_data/finalData.tsv", encoding="utf-8") as f:
        lines = f.read().splitlines()
    samples, words, labels = [], [], []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            words.append(parts[0])
            labels.append(parts[2])
        elif words:
            samples.append((words, labels))
            words, labels = [], []
    if words:
        samples.append((words, labels))

    rng = random.Random(0)
    rng.shuffle(samples)
    n_test = max(1, int(len(samples) * 0.2))
    train_all, test_samples = samples[n_test:], samples[:n_test]
    print(f"  {len(train_all)} train_all, {len(test_samples)} test")
    return train_all, test_samples


def make_val_split(train_samples, val_frac=0.1, seed=42):
    """Carve a validation set from training data."""
    rng = random.Random(seed)
    shuffled = train_samples.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]   # (train, val)


# ---------------------------------------------------------------------------
# Tokenisation and alignment helpers
# ---------------------------------------------------------------------------

def _tokenize_sample(words, labels, label2id, hing_tok, rob_tok, max_len):
    """Tokenize one (words, labels) sample with both tokenizers.

    Returns a dict with sub-token tensors, word_ids lists, and word-level
    label ids.  Truncation may reduce num_words if the sentence is long.
    """
    enc_kwargs = dict(
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    hing_enc = hing_tok(words, **enc_kwargs)
    rob_enc  = rob_tok(words,  **enc_kwargs)

    hing_wids = hing_enc.word_ids(0)   # list[int | None]
    rob_wids  = rob_enc.word_ids(0)

    # Actual word count after truncation (conservative: use the smaller)
    hing_max = max((w for w in hing_wids if w is not None), default=-1) + 1
    rob_max  = max((w for w in rob_wids  if w is not None), default=-1) + 1
    num_words = min(hing_max, rob_max, len(words))

    label_ids = [label2id[l] for l in labels[:num_words]]

    return {
        "hing_input_ids":      hing_enc.input_ids.squeeze(0),       # (max_len,)
        "hing_attention_mask": hing_enc.attention_mask.squeeze(0),
        "hing_word_ids":       hing_wids,                           # list[int|None]
        "rob_input_ids":       rob_enc.input_ids.squeeze(0),
        "rob_attention_mask":  rob_enc.attention_mask.squeeze(0),
        "rob_word_ids":        rob_wids,
        "label_ids":           label_ids,                           # list[int]
        "num_words":           num_words,
        "words":               words[:num_words],                   # for analysis
    }


# ---------------------------------------------------------------------------
# Dataset and collation
# ---------------------------------------------------------------------------

class CodeMixedDataset(Dataset):
    def __init__(self, samples, hing_tok, rob_tok, task, max_len=128):
        label_list = configs.TASK_LABELS[task]
        label2id   = {l: i for i, l in enumerate(label_list)}
        print(f"  Tokenising {len(samples)} samples ...", flush=True)
        self.data = [
            _tokenize_sample(w, l, label2id, hing_tok, rob_tok, max_len)
            for w, l in samples
        ]
        self.data = [d for d in self.data if d["num_words"] > 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    max_words = max(item["num_words"] for item in batch)

    labels_padded = []
    for item in batch:
        pad = [-100] * (max_words - len(item["label_ids"]))
        labels_padded.append(
            torch.tensor(item["label_ids"] + pad, dtype=torch.long)
        )

    return {
        "hing_input_ids":      torch.stack([b["hing_input_ids"]      for b in batch]),
        "hing_attention_mask": torch.stack([b["hing_attention_mask"] for b in batch]),
        "hing_word_ids":       [b["hing_word_ids"] for b in batch],
        "rob_input_ids":       torch.stack([b["rob_input_ids"]       for b in batch]),
        "rob_attention_mask":  torch.stack([b["rob_attention_mask"]  for b in batch]),
        "rob_word_ids":        [b["rob_word_ids"] for b in batch],
        "labels":              torch.stack(labels_padded),           # (B, max_words)
        "num_words":           torch.tensor([b["num_words"] for b in batch], dtype=torch.long),
        "words":               [b["words"] for b in batch],
    }


def make_dataloaders(train_s, val_s, test_s, hing_tok, rob_tok,
                     task, batch_size=32, max_len=128):
    """Build train / val / test DataLoaders."""
    kw = dict(collate_fn=collate_fn, num_workers=0, pin_memory=True)
    return (
        DataLoader(CodeMixedDataset(train_s, hing_tok, rob_tok, task, max_len),
                   batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(CodeMixedDataset(val_s,   hing_tok, rob_tok, task, max_len),
                   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(CodeMixedDataset(test_s,  hing_tok, rob_tok, task, max_len),
                   batch_size=batch_size, shuffle=False, **kw),
    )
