"""
Router tests:
  - MLP regression: moved RouterMLP matches pre-refactor reference (state_dict + output)
  - Shape/range tests for MLP, GRU, CNN
  - Hard-routing produces {0, 1} for all router types
  - sentence_level=True is rejected for GRU and CNN, accepted for MLP
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from routers import RouterMLP, RouterGRU, RouterCNN, build_router


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "router_mlp_reference.pt")


# ---------------------------------------------------------------------------
# Regression: moved RouterMLP reproduces saved reference exactly
# ---------------------------------------------------------------------------

def test_mlp_regression_matches_reference():
    ref = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    router = RouterMLP(input_dim=1536, hidden_dim=256, tau=1.0, hard_routing=False)
    router.load_state_dict(ref["state_dict"])
    router.eval()
    with torch.no_grad():
        out = router(ref["input"])
    assert out.shape == ref["expected_output"].shape
    assert torch.allclose(out, ref["expected_output"], atol=0.0), (
        "Moved RouterMLP produces different output than the pre-refactor "
        "reference. Checkpoint compatibility is broken."
    )


# ---------------------------------------------------------------------------
# Shape and range tests for all three routers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("router_type", ["mlp", "gru", "cnn"])
def test_shape_and_range(router_type):
    torch.manual_seed(0)
    router = build_router(router_type, input_dim=1536, tau=1.0, hard_routing=False)
    router.eval()

    B, T, D = 2, 10, 1536
    x = torch.randn(B, T, D)
    word_lens = torch.tensor([T, 7], dtype=torch.long)

    with torch.no_grad():
        alpha = router(x, word_lens=word_lens)

    assert alpha.shape == (B, T, 1), f"{router_type}: got {alpha.shape}"
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all(), (
        f"{router_type}: alpha out of [0,1]"
    )


@pytest.mark.parametrize("router_type", ["mlp", "gru", "cnn"])
def test_hard_routing_produces_binary(router_type):
    torch.manual_seed(0)
    router = build_router(router_type, input_dim=1536, tau=1.0, hard_routing=True)
    router.eval()

    B, T, D = 2, 10, 1536
    x = torch.randn(B, T, D)
    word_lens = torch.tensor([T, 7], dtype=torch.long)

    with torch.no_grad():
        alpha = router(x, word_lens=word_lens)

    vals = alpha.unique()
    assert set(vals.tolist()).issubset({0.0, 1.0}), (
        f"{router_type}: hard-routed α should be in {{0, 1}}, got {vals.tolist()}"
    )


# ---------------------------------------------------------------------------
# sentence_level restriction: MLP only
# ---------------------------------------------------------------------------

def test_sentence_level_accepted_for_mlp():
    build_router("mlp", input_dim=1536, tau=1.0, hard_routing=False, sentence_level=True)


@pytest.mark.parametrize("router_type", ["gru", "cnn"])
def test_sentence_level_rejected_for_sequence_routers(router_type):
    with pytest.raises(ValueError, match="sentence_level"):
        build_router(router_type, input_dim=1536, tau=1.0, hard_routing=False,
                     sentence_level=True)


# ---------------------------------------------------------------------------
# GRU requires word_lens; passing None raises ValueError
# ---------------------------------------------------------------------------

def test_gru_requires_word_lens():
    router = RouterGRU(input_dim=1536, tau=1.0, hard_routing=False)
    router.eval()
    x = torch.randn(2, 10, 1536)
    with pytest.raises(ValueError, match="word_lens"):
        router(x, word_lens=None)


# ---------------------------------------------------------------------------
# GRU respects word_lens: padded positions beyond length should have identical
# α across batch positions (all come from zero-initialised pad state).
# ---------------------------------------------------------------------------

def test_gru_pack_padded_ignores_padding():
    """Two samples of different lengths; the shorter one's real positions should
    not depend on what's in the padded positions."""
    torch.manual_seed(0)
    router = RouterGRU(
        input_dim=8, proj_dim=8, hidden_dim=4, tau=1.0, hard_routing=False,
    )
    router.eval()

    B, T, D = 2, 6, 8
    x = torch.randn(B, T, D)
    word_lens = torch.tensor([6, 3], dtype=torch.long)

    with torch.no_grad():
        alpha_a = router(x, word_lens=word_lens)

    # Change the padded-out region of sample 1 (positions 3..5)
    x2 = x.clone()
    x2[1, 3:, :] = torch.randn(T - 3, D) * 100  # drastic change in padding
    with torch.no_grad():
        alpha_b = router(x2, word_lens=word_lens)

    # The real positions of sample 1 (indices 0..2) must be unaffected
    assert torch.allclose(alpha_a[1, :3], alpha_b[1, :3], atol=1e-6), (
        "GRU output at real positions should not depend on padded-input contents."
    )
