"""
Router architectures for the MoE blend: scalar α ∈ [0,1] per word position.

All routers share the same interface:
    forward(x, word_lens=None) -> α
  where
    x:         (B, T, input_dim) for token-level, (B, input_dim) for sentence-level
    word_lens: (B,) long tensor of per-sample word counts (required by GRU,
               ignored by MLP and CNN)
    returns:   α with trailing singleton — (B, T, 1) or (B, 1)

Supports:
  - Temperature τ in the sigmoid: σ(logit / τ)
  - Hard routing via straight-through estimator at threshold 0.5 (R7-style)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# Shared post-logit activation
# ---------------------------------------------------------------------------

def _sigmoid_with_temperature_and_hard(
    logit: torch.Tensor,
    tau: float,
    hard_routing: bool,
    training: bool,
) -> torch.Tensor:
    """Apply σ(logit/τ) and optional straight-through hard routing."""
    alpha_soft = torch.sigmoid(logit / tau)
    if not hard_routing:
        return alpha_soft
    alpha_hard = (alpha_soft >= 0.5).to(alpha_soft.dtype)
    if training:
        return alpha_hard - alpha_soft.detach() + alpha_soft
    return alpha_hard


# ---------------------------------------------------------------------------
# MLP router (R1–R7)
# ---------------------------------------------------------------------------

class RouterMLP(nn.Module):
    """Per-word Linear → GELU → Linear → scalar α. No neighbour context."""

    def __init__(
        self,
        input_dim:    int   = 1536,
        hidden_dim:   int   = 256,
        tau:          float = 1.0,
        hard_routing: bool  = False,
    ):
        super().__init__()
        self.tau          = tau
        self.hard_routing = hard_routing
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.gelu    = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, word_lens=None) -> torch.Tensor:
        logit = self.linear2(self.gelu(self.linear1(x)))
        return _sigmoid_with_temperature_and_hard(
            logit, self.tau, self.hard_routing, self.training
        )


# ---------------------------------------------------------------------------
# GRU router (R8) — BiGRU over word-level sequence
# ---------------------------------------------------------------------------

class RouterGRU(nn.Module):
    """BiGRU router: Linear(proj) → pack → BiGRU → pad → Linear → α."""

    def __init__(
        self,
        input_dim:    int   = 1536,
        proj_dim:     int   = 128,
        hidden_dim:   int   = 64,
        tau:          float = 1.0,
        hard_routing: bool  = False,
    ):
        super().__init__()
        self.tau          = tau
        self.hard_routing = hard_routing
        self.proj = nn.Linear(input_dim, proj_dim)
        self.gru  = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor, word_lens: torch.Tensor = None) -> torch.Tensor:
        if word_lens is None:
            raise ValueError("RouterGRU.forward requires word_lens=(B,) tensor")
        B, T, _ = x.shape
        h = self.proj(x)   # (B, T, proj_dim)

        # pack_padded_sequence needs lengths on CPU and at least 1
        lens_cpu = word_lens.to("cpu").clamp(min=1)
        packed   = pack_padded_sequence(h, lens_cpu, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        # out: (B, T, 2*hidden_dim)
        logit = self.head(out)   # (B, T, 1)
        return _sigmoid_with_temperature_and_hard(
            logit, self.tau, self.hard_routing, self.training
        )


# ---------------------------------------------------------------------------
# CNN router (R9) — kernel=5, ±2-word window
# ---------------------------------------------------------------------------

class RouterCNN(nn.Module):
    """1D conv router: Conv1d(k=5, pad=2) → GELU → Conv1d(k=1) → α."""

    def __init__(
        self,
        input_dim:    int   = 1536,
        hidden_dim:   int   = 64,
        kernel_size:  int   = 5,
        tau:          float = 1.0,
        hard_routing: bool  = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        self.tau          = tau
        self.hard_routing = hard_routing
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=pad)
        self.gelu  = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, word_lens=None) -> torch.Tensor:
        # x: (B, T, input_dim) → Conv1d expects (B, C, T)
        h = x.transpose(1, 2)                 # (B, input_dim, T)
        h = self.gelu(self.conv1(h))          # (B, hidden_dim, T)
        logit = self.conv2(h).transpose(1, 2) # (B, T, 1)
        return _sigmoid_with_temperature_and_hard(
            logit, self.tau, self.hard_routing, self.training
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_router(
    router_type:    str,
    input_dim:      int,
    tau:            float,
    hard_routing:   bool,
    sentence_level: bool = False,
) -> nn.Module:
    if router_type == "mlp":
        return RouterMLP(input_dim, tau=tau, hard_routing=hard_routing)
    if sentence_level:
        raise ValueError(
            f"sentence_level=True is only supported with router_type='mlp', "
            f"got {router_type!r}"
        )
    if router_type == "gru":
        return RouterGRU(input_dim, tau=tau, hard_routing=hard_routing)
    if router_type == "cnn":
        return RouterCNN(input_dim, tau=tau, hard_routing=hard_routing)
    raise ValueError(f"Unknown router_type: {router_type!r}")
