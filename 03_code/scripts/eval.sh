#!/usr/bin/env bash
# Evaluate a checkpoint — pass checkpoint path as first argument
# Usage: bash eval.sh results/checkpoints/R2-lid-s42.pt
set -e
cd "$(dirname "$0")/../.."

CKPT="${1:-results/checkpoints/R2-lid-s42.pt}"
python main.py --mode eval --checkpoint "$CKPT"
