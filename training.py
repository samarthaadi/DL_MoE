"""
Trainer (training loop + early stopping + checkpointing) and evaluation metrics.
"""

import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import configs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, task: str, device, label_names: list) -> dict:
    """Run model on dataloader, return task-specific metric dict.

    LID  → weighted F1
    NER  → entity-level F1 (seqeval)
    POS  → accuracy
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = _to_device(batch, device)
            out   = model(batch)
            logits = out["logits"]          # (B, T, num_labels)
            labels = batch["labels"]        # (B, T)  — -100 for padding
            preds  = logits.argmax(-1)      # (B, T)

            for i in range(labels.shape[0]):
                n = batch["num_words"][i].item()
                all_preds.append(preds[i, :n].cpu().tolist())
                all_labels.append(labels[i, :n].cpu().tolist())

    if not all_preds:
        metric_name = configs.TASK_METRICS[task]
        return {"metric": 0.0, metric_name: 0.0}

    if task == "lid" or task == "pos":
        from sklearn.metrics import f1_score, accuracy_score

        flat_p = [p for seq in all_preds  for p in seq]
        flat_l = [l for seq in all_labels for l in seq]

        if task == "pos":
            metric = accuracy_score(flat_l, flat_p)
            return {"metric": metric, "accuracy": metric}
        else:
            metric = f1_score(flat_l, flat_p, average="weighted", zero_division=0)
            return {"metric": metric, "weighted_f1": metric}

    else:  # ner — seqeval entity-level F1
        from seqeval.metrics import f1_score as seq_f1

        # Convert ids → label strings; -100 should not appear (we slice to num_words)
        id2label = {i: l for i, l in enumerate(label_names)}
        str_preds  = [[id2label[p] for p in seq] for seq in all_preds]
        str_labels = [[id2label[l] for l in seq] for seq in all_labels]

        metric = seq_f1(str_labels, str_preds, zero_division=0)
        return {"metric": metric, "entity_f1": metric}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, train_dl, val_dl, task: str, device, cfg: dict = None):
        self.model    = model
        self.train_dl = train_dl
        self.val_dl   = val_dl
        self.task     = task
        self.device   = device

        cfg = cfg or {}
        self.lr         = cfg.get("lr",         configs.LR)
        self.max_epochs = cfg.get("max_epochs",  configs.MAX_EPOCHS)
        self.patience   = cfg.get("patience",    configs.PATIENCE)
        self.log_path   = cfg.get("log_path",    None)

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters found in model.")
        self.optimizer = AdamW(trainable, lr=self.lr)

        total_steps  = len(train_dl) * self.max_epochs
        warmup_steps = int(total_steps * configs.WARMUP_FRACTION)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ------------------------------------------------------------------ train

    def train(self) -> dict:
        """Train with early stopping. Returns best val metrics dict."""
        import csv
        best_metric, patience_cnt = -float("inf"), 0
        best_state      = None
        best_val_result = {"metric": 0.0}
        label_names = configs.TASK_LABELS[self.task]
        metric_name = configs.TASK_METRICS[self.task]
        epoch_log   = []

        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch()
            val_result = evaluate(
                self.model, self.val_dl, self.task, self.device, label_names
            )
            val_metric = val_result["metric"]
            epoch_log.append({"epoch": epoch + 1, "train_loss": train_loss,
                               "val_metric": val_metric})

            print(f"  Epoch {epoch+1:2d}/{self.max_epochs}  "
                  f"loss={train_loss:.4f}  "
                  f"val_{metric_name}={val_metric:.4f}", flush=True)

            if val_metric > best_metric:
                best_metric   = val_metric
                patience_cnt  = 0
                best_state    = _copy_trainable(self.model)
                best_val_result = val_result
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}.", flush=True)
                    break

        # Restore best weights
        if best_state is not None:
            _restore_trainable(self.model, best_state)

        # Write per-epoch CSV log
        if self.log_path and epoch_log:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_metric"])
                writer.writeheader()
                writer.writerows(epoch_log)

        return best_val_result

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_dl:
            batch = _to_device(batch, self.device)
            self.optimizer.zero_grad()

            out    = self.model(batch)
            logits = out["logits"]          # (B, T, num_labels)
            labels = batch["labels"]        # (B, T)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                configs.GRAD_CLIP,
            )
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(self.train_dl)

    # --------------------------------------------------------------- checkpoint

    def save_checkpoint(self, path: str, metrics: dict = None, exp_config: dict = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        trainable = {
            n: p.data.cpu().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        torch.save({
            "trainable":  trainable,
            "metrics":    metrics    or {},
            "exp_config": exp_config or {},
            "task":       self.task,
        }, path)
        print(f"  Checkpoint saved -> {path}", flush=True)

    @staticmethod
    def load_checkpoint(path: str, model) -> tuple:
        """Load trainable weights into model. Returns (metrics, exp_config)."""
        ckpt        = torch.load(path, map_location="cpu")
        model_params = dict(model.named_parameters())
        for name, data in ckpt["trainable"].items():
            if name in model_params and model_params[name].shape == data.shape:
                model_params[name].data.copy_(data)
            else:
                print(f"  Warning: skipping param {name!r} (shape mismatch or not found)")
        return ckpt.get("metrics", {}), ckpt.get("exp_config", {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _copy_trainable(model) -> dict:
    return {
        n: p.data.cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def _restore_trainable(model, state: dict):
    params = dict(model.named_parameters())
    for n, data in state.items():
        if n in params:
            params[n].data.copy_(data.to(params[n].device))
