#!/usr/bin/env bash
# Layer-wise freezing sweep (motivated by Yosinski et al. 2014): unfreeze progressively more
# residual stages of the pretrained ResNet-18. Same recipe as the main grid / low-data runs.
#   frozen (head only) and partial (layer4) are in results/lowdata.csv, full is in results/main_grid.csv,
#   so only the three intermediate modes are run here.
# Output: results/layerwise.csv   (3 modes x 4 sizes x 5 seeds = 60 runs; restart-safe)
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/lib.sh"
OUT=results/layerwise.csv
SEEDS5="${SEEDS5:-0 1 2 3 4}"
LR_LAYERWISE="${LR_LAYERWISE:-$LR_PRE_PARTIAL}"   # same LR as partial / full (1e-4)

for seed in $SEEDS5; do
  for pct in 1 2 5 10; do
    for mode in partial34 partial234 partial1234; do
      done_in $OUT pretrained $mode $pct $seed || run --init pretrained --mode $mode --pct $pct --seed $seed --lr $LR_LAYERWISE --imbalance $IMB --out $OUT --tag layerwise
    done
  done
done
