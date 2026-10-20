#!/usr/bin/env bash
# Step 5: learning-rate sweep. 10% subset, seed 0 only (cheap). Selection uses VALIDATION only.
# Output: results/lr_sweep.csv   (24 runs)
source "$(dirname "$0")/common.sh"
OUT=results/lr_sweep.csv

for lr in 1e-2 3e-3 1e-3 3e-4 1e-4 3e-5; do
  run --init scratch    --mode full    --pct 10 --seed 0 --lr $lr --imbalance none --out $OUT --tag lr-sweep
  run --init pretrained --mode full    --pct 10 --seed 0 --lr $lr --imbalance none --out $OUT --tag lr-sweep
  run --init pretrained --mode partial --pct 10 --seed 0 --lr $lr --imbalance none --out $OUT --tag lr-sweep
  run --init pretrained --mode frozen  --pct 10 --seed 0 --lr $lr --imbalance none --out $OUT --tag lr-sweep
done
