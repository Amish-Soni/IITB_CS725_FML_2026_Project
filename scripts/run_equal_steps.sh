#!/usr/bin/env bash
# Control experiment: scratch vs pretrained (full fine-tune) with an EQUAL number of optimiser
# steps per epoch at every data size (sampling with replacement), instead of one pass per epoch.
# Question: is scratch near chance at 1-2% because of too little data, or because it only got a
# handful of optimiser steps under the fixed-epoch recipe?
# Output: results/equal_steps.csv   (2 models x 4 sizes x 5 seeds = 40 runs; restart-safe)
# Budget: up to EQ_EPOCHS x EQ_STEPS = 50 x 30 = 1500 steps, patience 10 epochs (= 300 steps).
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/lib.sh"
OUT=results/equal_steps.csv
SEEDS5="${SEEDS5:-0 1 2 3 4}"
EQ_STEPS="${EQ_STEPS:-30}"
EPOCHS="${EQ_EPOCHS:-50}"
PATIENCE="${EQ_PATIENCE:-10}"

for seed in $SEEDS5; do
  for pct in 1 2 5 10; do
    done_in $OUT scratch full $pct $seed    || run --init scratch    --mode full --pct $pct --seed $seed --lr $LR_SCRATCH  --imbalance $IMB --steps-per-epoch $EQ_STEPS --out $OUT --tag equal-steps
    done_in $OUT pretrained full $pct $seed || run --init pretrained --mode full --pct $pct --seed $seed --lr $LR_PRE_FULL --imbalance $IMB --steps-per-epoch $EQ_STEPS --out $OUT --tag equal-steps
  done
done
