#!/usr/bin/env bash
# Train main MoE model (R2, best LID) on both tasks, both seeds
set -e
cd "$(dirname "$0")/../.."

python main.py --task lid --mode train --exp_id R2 --seed 42
python main.py --task lid --mode train --exp_id R2 --seed 123
python main.py --task pos --mode train --exp_id R2 --seed 42
python main.py --task pos --mode train --exp_id R2 --seed 123
