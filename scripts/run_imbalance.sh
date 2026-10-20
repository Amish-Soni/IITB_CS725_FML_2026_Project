#!/usr/bin/env bash
# Step 6: imbalance ablation (none vs weighted loss vs oversampling).
# Uses the LRs you chose in step 5 (set them via env vars or edit common.sh).
# Output: results/imbalance.csv   (2 inits x 2 sizes x 3 methods x 3 seeds = 36 runs)
source "$(dirname "$0")/common.sh"
OUT=results/imbalance.csv

for seed in $SEEDS; do
  for pct in 5 25; do
    for imb in none weighted oversample; do
      run --init scratch    --mode full --pct $pct --seed $seed --lr $LR_SCRATCH  --imbalance $imb --out $OUT --tag imbalance
      run --init pretrained --mode full --pct $pct --seed $seed --lr $LR_PRE_FULL --imbalance $imb --out $OUT --tag imbalance
    done
  done
done
