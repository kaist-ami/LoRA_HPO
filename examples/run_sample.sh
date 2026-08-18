#!/bin/bash
set -e
cd "$(dirname "$0")/.."
WANDB_MODE=offline python run_hpo.py \
    --config examples/config_sample.yaml \
    --task math \
    --dry_run \
    --seed 44 \
    --n_iters 5 \
    --group sample
