"""
One-time setup: download all models and datasets into models_and_data/.

Usage: py setup_data.py
Skips anything that already exists.
"""

import os
import urllib.request

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

os.makedirs("models_and_data/comi_lingua", exist_ok=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

for hf_name, local_path in [
    ("l3cube-pune/hing-bert", "models_and_data/hing-bert"),
    ("roberta-base",          "models_and_data/roberta-base"),
]:
    if os.path.exists(local_path):
        print(f"  {local_path} already exists, skipping")
    else:
        print(f"Downloading {hf_name} ...")
        AutoTokenizer.from_pretrained(hf_name).save_pretrained(local_path)
        AutoModel.from_pretrained(hf_name).save_pretrained(local_path)
        print(f"  Saved -> {local_path}")

# ---------------------------------------------------------------------------
# COMI-LINGUA (HuggingFace dataset)
# ---------------------------------------------------------------------------

for task in ["LID", "POS", "NER"]:
    path = f"models_and_data/comi_lingua/{task}"
    if os.path.exists(path):
        print(f"  {path} already exists, skipping")
    else:
        print(f"Downloading COMI-LINGUA/{task} ...")
        ds = load_dataset("LingoIITGN/COMI-LINGUA", task)
        ds.save_to_disk(path)
        print(f"  Saved -> {path}")

# ---------------------------------------------------------------------------
# SilentFlame NER + Twitter POS (raw files)
# ---------------------------------------------------------------------------

for url, dest in [
    (
        "https://raw.githubusercontent.com/SilentFlame/"
        "Named-Entity-Recognition/master/Twitterdata/annotatedData.csv",
        "models_and_data/annotatedData.csv",
    ),
    (
        "https://raw.githubusercontent.com/soicalnlpataclAnon/"
        "A-Twitter-Hindi-English-Code-Mixed-Dataset-for-POS-Tagging"
        "/master/finalData.tsv",
        "models_and_data/finalData.tsv",
    ),
]:
    if os.path.exists(dest):
        print(f"  {dest} already exists, skipping")
    else:
        print(f"Downloading {dest} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved -> {dest}")

print("\nDone. All assets are in models_and_data/")
