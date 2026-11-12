#!/usr/bin/env bash
# Re-train a few configurations and SAVE val + test probabilities, to diagnose why validation
# balanced accuracy (~0.96) is much higher than test (~0.85-0.89).
# Output: results/preds/*.csv and results/diagnose_runs.csv  (3 configs x 3 seeds = 9 runs)
# These are repeats of runs that already exist (same seed/subset); they are NOT used in the
# main results tables.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/lib.sh"
OUT=results/diagnose_runs.csv
DIAG_SEEDS="${DIAG_SEEDS:-0 1 2}"

for seed in $DIAG_SEEDS; do
  for cfg in "pretrained full 10 $LR_PRE_FULL" "pretrained full 100 $LR_PRE_FULL" "scratch full 100 $LR_SCRATCH"; do
    set -- $cfg
    done_in $OUT $1 $2 $3 $seed || run --init $1 --mode $2 --pct $3 --seed $seed --lr $4 --imbalance $IMB --out $OUT --tag diagnose --save-preds
  done
done
