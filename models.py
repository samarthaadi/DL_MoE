"""
Model definitions: frozen experts, full MoE model, baselines.

Router architectures live in routers.py (MLP, GRU, CNN).
All expert parameters are frozen by default (requires_grad=False).
Only the router and task head receive gradients.
Exception: B1 baseline (SingleExpertModel with frozen=False).
"""

import torch
import torch.nn as nn
from transformers import AutoModel

import configs
from routers import build_router


# ---------------------------------------------------------------------------
# Word alignment (also imported by tests)
# ---------------------------------------------------------------------------

def align_subtokens_to_words(
    hidden: torch.Tensor,
    word_ids: list,
    max_words: int,
) -> torch.Tensor:
    """Average sub-token hidden states to word-level representations.

    Args:
        hidden:    (seq_len, D) — single sample from one expert
        word_ids:  list[int | None] from tokenizer.word_ids()
        max_words: output sequence length (pad remainder with zeros)

    Returns:
        (max_words, D)
    """
    D = hidden.shape[-1]
    out   = torch.zeros(max_words, D,    device=hidden.device, dtype=hidden.dtype)
    count = torch.zeros(max_words,       device=hidden.device, dtype=hidden.dtype)

    for i, wid in enumerate(word_ids):
        if wid is None or wid >= max_words:
            continue
        out[wid]   += hidden[i]
        count[wid] += 1

    nz = count > 0
    out[nz] /= count[nz, None]   # average; zero-padded words stay zero
    return out


# ---------------------------------------------------------------------------
# Frozen expert wrapper
# ---------------------------------------------------------------------------

class FrozenExpert(nn.Module):
    def __init__(self, model_name: str, frozen: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        """Returns last hidden states: (batch, seq_len, 768)."""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


# ---------------------------------------------------------------------------
# Full MoE model (R1–R9, A2–A3)
# ---------------------------------------------------------------------------

class MoEModel(nn.Module):
    def __init__(
        self,
        num_labels:     int,
        tau:            float = 1.0,
        router_input:   str   = "both",   # "both" | "hing" | "rob"
        sentence_level: bool  = False,
        hard_routing:   bool  = False,
        router_type:    str   = "mlp",    # "mlp" | "gru" | "cnn"
    ):
        super().__init__()
        self.router_input   = router_input
        self.sentence_level = sentence_level

        self.hing_expert = FrozenExpert(configs.HINGBERT, frozen=True)
        self.rob_expert  = FrozenExpert(configs.ROBERTA,  frozen=True)

        # Router input dim depends on ablation setting
        if sentence_level:
            router_dim = 1536   # always uses both CLS reps
        elif router_input == "both":
            router_dim = 1536
        else:
            router_dim = 768

        self.router = build_router(
            router_type, router_dim,
            tau=tau, hard_routing=hard_routing, sentence_level=sentence_level,
        )
        self.task_head = nn.Linear(768, num_labels)

    def forward(self, batch: dict) -> dict:
        B         = batch["hing_input_ids"].shape[0]
        max_words = batch["labels"].shape[1]

        # ---- Expert forward passes (no gradient through experts) -----------
        with torch.no_grad():
            h_hing_sub = self.hing_expert(
                batch["hing_input_ids"], batch["hing_attention_mask"]
            )   # (B, seq_len, 768)
            h_rob_sub  = self.rob_expert(
                batch["rob_input_ids"],  batch["rob_attention_mask"]
            )

        # ---- Word-level alignment ------------------------------------------
        h_hing = torch.stack([
            align_subtokens_to_words(h_hing_sub[i], batch["hing_word_ids"][i], max_words)
            for i in range(B)
        ])   # (B, max_words, 768)
        h_rob = torch.stack([
            align_subtokens_to_words(h_rob_sub[i],  batch["rob_word_ids"][i],  max_words)
            for i in range(B)
        ])

        # ---- Router ----------------------------------------------------------
        word_lens = batch["num_words"]   # (B,) — used by sequence routers (GRU)
        if self.sentence_level:
            # Use CLS (position 0) hidden states from both experts
            cls_combined = torch.cat(
                [h_hing_sub[:, 0, :], h_rob_sub[:, 0, :]], dim=-1
            )   # (B, 1536)
            alpha = self.router(cls_combined)               # (B, 1)
            alpha = alpha.unsqueeze(1).expand(B, max_words, 1)
        else:
            if self.router_input == "hing":
                router_in = h_hing
            elif self.router_input == "rob":
                router_in = h_rob
            else:
                router_in = torch.cat([h_hing, h_rob], dim=-1)   # (B, T, 1536)
            alpha = self.router(router_in, word_lens=word_lens)   # (B, T, 1)

        # ---- Blend + task head ---------------------------------------------
        blended = alpha * h_hing + (1 - alpha) * h_rob   # (B, T, 768)
        logits  = self.task_head(blended)                  # (B, T, num_labels)

        l2_disagreement = torch.norm(h_hing - h_rob, dim=-1)   # (B, T)

        return {
            "logits":          logits,
            "alpha":           alpha.squeeze(-1),       # (B, T)
            "l2_disagreement": l2_disagreement,
        }


# ---------------------------------------------------------------------------
# Single-expert baselines (B1, B2, B3)
# ---------------------------------------------------------------------------

class SingleExpertModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int, frozen: bool = True):
        super().__init__()
        self.which  = "hing" if "hing" in model_name.lower() else "rob"
        self.frozen = frozen
        self.expert    = FrozenExpert(model_name, frozen=frozen)
        self.task_head = nn.Linear(768, num_labels)

    def forward(self, batch: dict) -> dict:
        B         = batch["hing_input_ids"].shape[0]
        max_words = batch["labels"].shape[1]

        input_ids  = batch[f"{self.which}_input_ids"]
        attn_mask  = batch[f"{self.which}_attention_mask"]
        word_ids   = batch[f"{self.which}_word_ids"]

        if self.frozen:
            with torch.no_grad():
                h_sub = self.expert(input_ids, attn_mask)
        else:
            h_sub = self.expert(input_ids, attn_mask)

        h_words = torch.stack([
            align_subtokens_to_words(h_sub[i], word_ids[i], max_words)
            for i in range(B)
        ])
        logits = self.task_head(h_words)
        return {"logits": logits, "alpha": None, "l2_disagreement": None}


# ---------------------------------------------------------------------------
# Fixed 50/50 baseline (B4)
# ---------------------------------------------------------------------------

class FixedAverageModel(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.hing_expert = FrozenExpert(configs.HINGBERT, frozen=True)
        self.rob_expert  = FrozenExpert(configs.ROBERTA,  frozen=True)
        self.task_head   = nn.Linear(768, num_labels)

    def forward(self, batch: dict) -> dict:
        B         = batch["hing_input_ids"].shape[0]
        max_words = batch["labels"].shape[1]

        with torch.no_grad():
            h_hing_sub = self.hing_expert(
                batch["hing_input_ids"], batch["hing_attention_mask"]
            )
            h_rob_sub  = self.rob_expert(
                batch["rob_input_ids"],  batch["rob_attention_mask"]
            )

        h_hing = torch.stack([
            align_subtokens_to_words(h_hing_sub[i], batch["hing_word_ids"][i], max_words)
            for i in range(B)
        ])
        h_rob = torch.stack([
            align_subtokens_to_words(h_rob_sub[i],  batch["rob_word_ids"][i],  max_words)
            for i in range(B)
        ])

        blended = 0.5 * h_hing + 0.5 * h_rob
        logits  = self.task_head(blended)
        alpha   = torch.full((B, max_words), 0.5, device=blended.device)

        return {"logits": logits, "alpha": alpha, "l2_disagreement": None}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(exp_config: dict, task: str, device: str = "cpu") -> nn.Module:
    num_labels  = len(configs.TASK_LABELS[task])
    model_mode  = exp_config.get("model_mode", "moe")

    if model_mode == "moe":
        model = MoEModel(
            num_labels     = num_labels,
            tau            = exp_config.get("tau",            1.0),
            router_input   = exp_config.get("router_input",   "both"),
            sentence_level = exp_config.get("sentence_level", False),
            hard_routing   = exp_config.get("hard_routing",   False),
            router_type    = exp_config.get("router_type",    "mlp"),
        )
    elif model_mode == "hingbert":
        model = SingleExpertModel(
            configs.HINGBERT, num_labels,
            frozen=exp_config.get("frozen", True),
        )
    elif model_mode == "roberta":
        model = SingleExpertModel(configs.ROBERTA, num_labels, frozen=True)
    elif model_mode == "fixed_avg":
        model = FixedAverageModel(num_labels)
    else:
        raise ValueError(f"Unknown model_mode: {model_mode!r}")

    return model.to(device)
