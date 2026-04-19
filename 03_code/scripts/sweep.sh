#!/usr/bin/env bash
# Run the full experiment sweep (all 17 configs × 2 tasks × 2 seeds)
# Skips runs whose checkpoints already exist.
set -e
cd "$(dirname "$0")/../.."

python main.py --mode sweep
