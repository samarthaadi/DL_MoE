"""
CLI entry point for training, evaluation, interpretability analysis, and sweeps.

Usage examples:
  python main.py --task lid --mode train --tau 1.0 --seed 42
  python main.py --task lid --mode eval  --checkpoint results/checkpoints/R1-lid-s42.pt
  python main.py --task lid --mode analysis --checkpoint results/checkpoints/R1-lid-s42.pt
  python main.py --task lid --mode train --model_mode hingbert --seed 42
  python main.py --mode sweep
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer

import configs
from analysis import analysis_cross_task, run_analyses, run_cross_task_analyses
from data import load_comi_lingua, load_ner_data, load_pos_data, make_dataloaders, make_val_split
from models import build_model
from training import Trainer, evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def ckpt_path(exp_id: str, task: str, seed: int) -> str:
    return f"results/checkpoints/{exp_id}-{task}-s{seed}.pt"


def metrics_path(exp_id: str, task: str, seed: int) -> str:
    return f"results/metrics/{exp_id}-{task}-s{seed}.json"


def log_path(exp_id: str, task: str, seed: int) -> str:
    return f"results/logs/{exp_id}-{task}-s{seed}.csv"


def _exp_config_from_args(args) -> dict:
    """Build an experiment config dict from CLI args."""
    cfg = {"model_mode": args.model_mode}
    if args.model_mode == "moe":
        cfg["tau"]            = args.tau
        cfg["router_input"]   = args.router_input
        cfg["sentence_level"] = args.sentence_level
        cfg["hard_routing"]   = args.hard_routing
        cfg["router_type"]    = args.router_type
    if args.model_mode == "hingbert":
        cfg["frozen"] = not args.no_frozen
    return cfg


def _load_tokenizers():
    print("Loading tokenizers ...")
    hing_tok = AutoTokenizer.from_pretrained(configs.HINGBERT)
    rob_tok  = AutoTokenizer.from_pretrained(configs.ROBERTA, add_prefix_space=True)
    return hing_tok, rob_tok


def _load_data(task: str, seed: int, batch_size: int, hing_tok, rob_tok):
    if task == "lid":
        train_all, test_samples = load_comi_lingua(task)
    elif task == "ner":
        train_all, test_samples = load_ner_data()
    else:
        train_all, test_samples = load_pos_data()
    train_samples, val_samples = make_val_split(train_all, seed=seed)
    print(f"  Split sizes: train={len(train_samples)}, "
          f"val={len(val_samples)}, test={len(test_samples)}")
    train_dl, val_dl, test_dl = make_dataloaders(
        train_samples, val_samples, test_samples,
        hing_tok, rob_tok,
        task=task,
        batch_size=batch_size,
        max_len=configs.MAX_SEQ_LEN,
    )
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_train(args):
    set_seed(args.seed)
    device     = get_device()
    exp_config = _exp_config_from_args(args)
    exp_id     = args.exp_id or "custom"

    print(f"\n=== TRAIN  task={args.task}  exp={exp_id}  seed={args.seed}  device={device} ===")
    hing_tok, rob_tok = _load_tokenizers()
    train_dl, val_dl, test_dl = _load_data(
        args.task, args.seed, args.batch_size, hing_tok, rob_tok
    )

    model       = build_model(exp_config, args.task, device)
    trainer_cfg = {"log_path": log_path(exp_id, args.task, args.seed)}
    if args.max_epochs:
        trainer_cfg["max_epochs"] = args.max_epochs
    trainer = Trainer(model, train_dl, val_dl, args.task, device, cfg=trainer_cfg)

    print(f"Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_val = trainer.train()

    # Test evaluation
    test_result = evaluate(
        model, test_dl, args.task, device, configs.TASK_LABELS[args.task]
    )
    print(f"\nTest result: {test_result}")

    # Save checkpoint and metrics
    path  = ckpt_path(exp_id, args.task, args.seed)
    mpath = metrics_path(exp_id, args.task, args.seed)
    trainer.save_checkpoint(path, metrics=test_result, exp_config=exp_config)
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "w") as f:
        json.dump({"val": best_val, "test": test_result, "config": exp_config}, f, indent=2)

    return test_result


def run_eval(args):
    device = get_device()
    print(f"\n=== EVAL  checkpoint={args.checkpoint} ===")

    # Load checkpoint metadata to reconstruct model
    ckpt_data  = torch.load(args.checkpoint, map_location="cpu")
    exp_config = ckpt_data.get("exp_config", {})
    task       = args.task or ckpt_data.get("task", "lid")

    if not exp_config:
        print("  Warning: no exp_config in checkpoint — using CLI args for model config.")
        exp_config = _exp_config_from_args(args)

    hing_tok, rob_tok = _load_tokenizers()
    _, _, test_dl = _load_data(task, seed=42, batch_size=args.batch_size,
                               hing_tok=hing_tok, rob_tok=rob_tok)

    model = build_model(exp_config, task, device)
    Trainer.load_checkpoint(args.checkpoint, model)
    model.eval()

    result = evaluate(model, test_dl, task, device, configs.TASK_LABELS[task])
    print(f"Test result: {result}")
    return result


def run_analysis(args):
    """Run all five analyses for one checkpoint.

    After individual analyses are done, checks results/figures/ for records
    from other tasks and runs cross-task analysis (Analysis 5) if both
    LID and POS records are present.
    """
    device = get_device()
    print(f"\n=== ANALYSIS  checkpoint={args.checkpoint} ===")

    ckpt_data  = torch.load(args.checkpoint, map_location="cpu")
    exp_config = ckpt_data.get("exp_config", {})
    task       = args.task or ckpt_data.get("task", "lid")
    exp_id     = args.exp_id or "custom"

    if not exp_config:
        exp_config = _exp_config_from_args(args)

    hing_tok, rob_tok = _load_tokenizers()
    _, _, test_dl = _load_data(task, seed=42, batch_size=args.batch_size,
                               hing_tok=hing_tok, rob_tok=rob_tok)

    model = build_model(exp_config, task, device)
    Trainer.load_checkpoint(args.checkpoint, model)
    model.eval()

    import json as _json
    figures_dir = "results/figures"
    stats = run_analyses(model, test_dl, task, device,
                         figures_dir=figures_dir, exp_id=exp_id)

    # Cross-task analysis: try to pair with records from the other task
    other_task = "pos" if task == "lid" else "lid"
    other_records_path = os.path.join(
        figures_dir, f"{exp_id}_{other_task}_records.json"
    )
    if os.path.exists(other_records_path):
        print(f"\nFound {other_task} records — running cross-task analysis ...")
        with open(other_records_path) as f:
            other_records = _json.load(f)
        current_records_path = os.path.join(
            figures_dir, f"{exp_id}_{task}_records.json"
        )
        with open(current_records_path) as f:
            current_records = _json.load(f)
        records_per_task = {task: current_records, other_task: other_records}
        run_cross_task_analyses(records_per_task, figures_dir)
    else:
        print(f"\nNo {other_task} records found for {exp_id} — "
              f"run analysis on a {other_task} checkpoint with --exp_id {exp_id} "
              f"to enable cross-task comparison.")


def run_sweep(args):
    """Run all experiments × tasks × seeds. Skips already-completed runs."""
    print("\n=== SWEEP ===")
    device    = get_device()
    hing_tok, rob_tok = _load_tokenizers()

    all_results = {}   # {exp_id: {task: [metric_per_seed]}}

    experiments = {k: v for k, v in configs.EXPERIMENTS.items()
                   if not args.exp_ids or k in args.exp_ids}

    for exp_id, exp_config in experiments.items():
        for task in configs.TASKS:
            seed_metrics = []
            for seed in configs.SEEDS:
                cp   = ckpt_path(exp_id, task, seed)
                mp   = metrics_path(exp_id, task, seed)

                if os.path.exists(cp):
                    print(f"  Skipping {exp_id}/{task}/s{seed} (checkpoint exists)")
                    if os.path.exists(mp):
                        with open(mp) as f:
                            saved = json.load(f)
                        seed_metrics.append(saved["test"]["metric"])
                    continue

                print(f"\n--- {exp_id} / {task} / seed={seed} ---")
                set_seed(seed)
                train_dl, val_dl, test_dl = _load_data(
                    task, seed, args.batch_size, hing_tok, rob_tok
                )
                model   = build_model(exp_config, task, device)
                trainer = Trainer(model, train_dl, val_dl, task, device,
                                  cfg={"log_path": log_path(exp_id, task, seed)})
                trainer.train()

                test_result = evaluate(
                    model, test_dl, task, device, configs.TASK_LABELS[task]
                )
                print(f"  Test: {test_result}")

                os.makedirs(os.path.dirname(cp), exist_ok=True)
                os.makedirs(os.path.dirname(mp), exist_ok=True)
                trainer.save_checkpoint(cp, metrics=test_result, exp_config=exp_config)
                with open(mp, "w") as f:
                    json.dump({"test": test_result, "config": exp_config}, f, indent=2)

                seed_metrics.append(test_result["metric"])

            if seed_metrics:
                mean = float(np.mean(seed_metrics))
                std  = float(np.std(seed_metrics))
                key  = f"{exp_id}/{task}"
                all_results[key] = {"mean": mean, "std": std, "seeds": seed_metrics}
                print(f"  {key}: {mean:.4f} ± {std:.4f}")

    # Save aggregated results
    agg_path = "results/metrics/aggregated.json"
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated results saved -> {agg_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Frozen Expert MoE for code-mixed NLP")

    p.add_argument("--mode",  choices=["train", "eval", "analysis", "sweep"],
                   default="train")
    p.add_argument("--task",  choices=["lid", "ner", "pos"], default=None)
    p.add_argument("--seed",  type=int, default=42)

    # Model config
    p.add_argument("--model_mode",
                   choices=["moe", "hingbert", "roberta", "fixed_avg"],
                   default="moe")
    p.add_argument("--tau",           type=float, default=1.0)
    p.add_argument("--router_input",  choices=["both", "hing", "rob"], default="both")
    p.add_argument("--router_type",   choices=["mlp", "gru", "cnn"],  default="mlp")
    p.add_argument("--sentence_level", action="store_true")
    p.add_argument("--hard_routing",   action="store_true")
    p.add_argument("--no_frozen",      action="store_true",
                   help="Unfreeze expert (B1 baseline only)")

    # Experiment ID (used for checkpoint naming and sweep lookup)
    p.add_argument("--exp_id", default=None,
                   help="Experiment ID from configs.EXPERIMENTS (e.g. R1, B2). "
                        "If given, overrides model config flags.")

    # Paths
    p.add_argument("--checkpoint", default=None)

    # Training
    p.add_argument("--exp_ids", nargs="*", default=None,
                   help="Subset of experiment IDs to run in sweep (e.g. B2 B3 R1)")
    p.add_argument("--batch_size", type=int, default=configs.BATCH_SIZE)
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Override MAX_EPOCHS (e.g. 3 for a quick sanity check)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # If --exp_id given, override model config from the registry
    if args.exp_id and args.exp_id in configs.EXPERIMENTS:
        ec = configs.EXPERIMENTS[args.exp_id]
        args.model_mode     = ec.get("model_mode",    args.model_mode)
        args.tau            = ec.get("tau",            args.tau)
        args.router_input   = ec.get("router_input",   args.router_input)
        args.router_type    = ec.get("router_type",    args.router_type)
        args.sentence_level = ec.get("sentence_level", args.sentence_level)
        args.hard_routing   = ec.get("hard_routing",   args.hard_routing)
        args.no_frozen      = not ec.get("frozen",     True)

    if args.mode == "sweep":
        run_sweep(args)
    elif args.mode == "train":
        if args.task is None:
            raise ValueError("--task is required for train mode")
        run_train(args)
    elif args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for eval mode")
        run_eval(args)
    elif args.mode == "analysis":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for analysis mode")
        run_analysis(args)


if __name__ == "__main__":
    main()
