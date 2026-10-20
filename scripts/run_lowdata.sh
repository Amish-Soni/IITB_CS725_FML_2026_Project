#!/usr/bin/env bash
# Step 8: low-data deep dive: frozen / partial fine-tune of the pretrained model.
# (Full fine-tune and scratch at these sizes are already in results/main_grid.csv.)
# More epochs + patience here: at 1% there is only ~1 batch per epoch.
# Output: results/lowdata.csv   (2 modes x 4 sizes x 3 seeds = 24 runs)
source "$(dirname "$0")/common.sh"
OUT=results/lowdata.csv
EPOCHS="${LOWDATA_EPOCHS:-60}"
PATIENCE="${LOWDATA_PATIENCE:-12}"

for seed in $SEEDS; do
  for pct in 1 2 5 10; do
    run --init pretrained --mode frozen  --pct $pct --seed $seed --lr $LR_PRE_FROZEN   --imbalance $IMB --out $OUT --tag lowdata
    run --init pretrained --mode partial --pct $pct --seed $seed --lr $LR_PRE_PARTIAL  --imbalance $IMB --out $OUT --tag lowdata
  done
done
