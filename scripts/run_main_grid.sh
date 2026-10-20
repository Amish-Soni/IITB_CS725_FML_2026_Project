#!/usr/bin/env bash
# Step 7: main data-starvation grid, scratch vs pretrained (full fine-tune).
# Output: results/main_grid.csv   (2 inits x 7 sizes x 3 seeds = 42 runs)
# Safe to re-run: combos already present in the CSV are skipped.
source "$(dirname "$0")/common.sh"
OUT=results/main_grid.csv

done_already() {  # init mode pct seed
  [ -f "$OUT" ] && python - "$OUT" "$@" <<'EOF'
import sys, pandas as pd
f, init, mode, pct, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
d = pd.read_csv(f)
sys.exit(0 if ((d.init==init)&(d["mode"]==mode)&(d.pct==pct)&(d.seed==seed)).any() else 1)
EOF
}

for seed in $SEEDS; do
  for pct in 1 2 5 10 25 50 100; do
    done_already scratch full $pct $seed    || run --init scratch    --mode full --pct $pct --seed $seed --lr $LR_SCRATCH  --imbalance $IMB --out $OUT --tag main
    done_already pretrained full $pct $seed || run --init pretrained --mode full --pct $pct --seed $seed --lr $LR_PRE_FULL --imbalance $IMB --out $OUT --tag main
  done
done
