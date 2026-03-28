"""
Unit tests for word-level alignment logic.

Run BEFORE any experiments:
    python -m pytest tests/ -v

These tests verify that:
  1. Both tokenizers produce the same word count after alignment
  2. Special tokens (CLS/SEP, <s>/</s>) are correctly identified as None in word_ids()
  3. align_subtokens_to_words correctly averages sub-tokens and pads
  4. Alignment is consistent for English, romanized Hindi, and mixed sentences
  5. Truncation produces consistent word counts from both tokenizers

NOTE: First run downloads HingBERT (~440MB) and RoBERTa (~500MB).
"""

import pytest
import torch
from transformers import AutoTokenizer

# The function under test lives in models.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import align_subtokens_to_words
import configs


# ---------------------------------------------------------------------------
# Fixtures — loaded once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hing_tok():
    return AutoTokenizer.from_pretrained(configs.HINGBERT)


@pytest.fixture(scope="session")
def rob_tok():
    return AutoTokenizer.from_pretrained(configs.ROBERTA, add_prefix_space=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def word_count_after_alignment(tok, words, max_len=128):
    enc  = tok(words, is_split_into_words=True, truncation=True,
                max_length=max_len, padding="max_length", return_tensors="pt")
    wids = enc.word_ids(0)
    return max((w for w in wids if w is not None), default=-1) + 1


# ---------------------------------------------------------------------------
# Test 1: Special tokens → None in word_ids
# ---------------------------------------------------------------------------

def test_special_tokens_are_none_hingbert(hing_tok):
    words = ["hello", "world"]
    enc   = hing_tok(words, is_split_into_words=True, return_tensors="pt")
    wids  = enc.word_ids(0)
    assert wids[0]  is None, "HingBERT [CLS] token should have word_id=None"
    assert wids[-1] is None, "HingBERT [SEP] token should have word_id=None"


def test_special_tokens_are_none_roberta(rob_tok):
    words = ["hello", "world"]
    enc   = rob_tok(words, is_split_into_words=True, return_tensors="pt")
    wids  = enc.word_ids(0)
    assert wids[0]  is None, "RoBERTa <s> token should have word_id=None"
    assert wids[-1] is None, "RoBERTa </s> token should have word_id=None"


# ---------------------------------------------------------------------------
# Test 2: Same word count — English sentence
# ---------------------------------------------------------------------------

def test_same_word_count_english(hing_tok, rob_tok):
    words = ["this", "is", "a", "simple", "English", "test"]
    h = word_count_after_alignment(hing_tok, words)
    r = word_count_after_alignment(rob_tok,  words)
    assert h == r == len(words), (
        f"Word counts diverge: hing={h}, rob={r}, expected={len(words)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Same word count — romanized Hindi
# ---------------------------------------------------------------------------

def test_same_word_count_hindi_roman(hing_tok, rob_tok):
    words = ["main", "gym", "ja", "raha", "hoon", "aaj"]
    h = word_count_after_alignment(hing_tok, words)
    r = word_count_after_alignment(rob_tok,  words)
    assert h == r == len(words), (
        f"Word counts diverge: hing={h}, rob={r}, expected={len(words)}"
    )


# ---------------------------------------------------------------------------
# Test 4: Same word count — code-mixed sentence
# ---------------------------------------------------------------------------

def test_same_word_count_code_mixed(hing_tok, rob_tok):
    words = ["aaj", "ka", "weather", "bahut", "nice", "hai"]
    h = word_count_after_alignment(hing_tok, words)
    r = word_count_after_alignment(rob_tok,  words)
    assert h == r, f"Word counts diverge: hing={h}, rob={r}"


# ---------------------------------------------------------------------------
# Test 5: Single-word input
# ---------------------------------------------------------------------------

def test_single_word(hing_tok, rob_tok):
    words = ["hello"]
    h = word_count_after_alignment(hing_tok, words)
    r = word_count_after_alignment(rob_tok,  words)
    assert h == 1 and r == 1


# ---------------------------------------------------------------------------
# Test 6: align_subtokens_to_words — correct averaging
# ---------------------------------------------------------------------------

def test_align_averages_subtokens():
    """Word 0 has 2 sub-tokens; word 1 has 1. Verify averaging."""
    D      = 4
    hidden = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],   # index 0 → None (CLS)
        [1.0, 2.0, 3.0, 4.0],   # index 1 → word 0
        [3.0, 4.0, 5.0, 6.0],   # index 2 → word 0
        [7.0, 8.0, 9.0, 0.0],   # index 3 → word 1
        [0.0, 0.0, 0.0, 0.0],   # index 4 → None (SEP)
    ], dtype=torch.float)
    word_ids = [None, 0, 0, 1, None]

    aligned = align_subtokens_to_words(hidden, word_ids, max_words=2)

    assert aligned.shape == (2, D)
    expected_w0 = (hidden[1] + hidden[2]) / 2
    assert torch.allclose(aligned[0], expected_w0), \
        f"Word 0 avg wrong: {aligned[0]} vs {expected_w0}"
    assert torch.allclose(aligned[1], hidden[3]), \
        f"Word 1 wrong: {aligned[1]} vs {hidden[3]}"


# ---------------------------------------------------------------------------
# Test 7: align_subtokens_to_words — padding with zeros
# ---------------------------------------------------------------------------

def test_align_pads_with_zeros():
    hidden   = torch.randn(3, 8)
    word_ids = [None, 0, None]   # only one actual word

    aligned = align_subtokens_to_words(hidden, word_ids, max_words=5)

    assert aligned.shape == (5, 8)
    assert torch.allclose(aligned[1:], torch.zeros(4, 8)), \
        "Padding positions should be zero"
    assert torch.allclose(aligned[0], hidden[1]), \
        "Word 0 should equal hidden[1]"


# ---------------------------------------------------------------------------
# Test 8: No information leakage from None positions
# ---------------------------------------------------------------------------

def test_align_ignores_none_positions():
    hidden   = torch.ones(5, 2)
    hidden[0] = 99.0   # CLS — should be ignored
    hidden[4] = 99.0   # SEP — should be ignored
    word_ids  = [None, 0, 1, 2, None]

    aligned = align_subtokens_to_words(hidden, word_ids, max_words=3)

    for i in range(3):
        assert torch.allclose(aligned[i], torch.ones(2)), \
            f"Word {i} should be 1.0 but got {aligned[i]}"


# ---------------------------------------------------------------------------
# Test 9: Long sentence truncation — counts stay consistent
# ---------------------------------------------------------------------------

def test_truncation_consistent(hing_tok, rob_tok):
    """A 200-word sentence will be truncated. Both tokenizers must still agree."""
    words = ["word"] * 200
    h = word_count_after_alignment(hing_tok, words, max_len=64)
    r = word_count_after_alignment(rob_tok,  words, max_len=64)
    assert h == r, f"After truncation, word counts differ: hing={h}, rob={r}"
    assert h < len(words), "Truncation should reduce word count"


# ---------------------------------------------------------------------------
# Test 10: Integration — is_split_into_words=True preserves word boundaries
# ---------------------------------------------------------------------------

def test_word_boundary_preservation(hing_tok, rob_tok):
    """Every word in the input should appear at least once in word_ids."""
    words = ["the", "quick", "brown", "fox", "jumps"]
    for tok in (hing_tok, rob_tok):
        enc  = tok(words, is_split_into_words=True, return_tensors="pt")
        wids = set(w for w in enc.word_ids(0) if w is not None)
        assert wids == set(range(len(words))), \
            f"{tok.__class__.__name__}: missing word indices {set(range(len(words))) - wids}"
