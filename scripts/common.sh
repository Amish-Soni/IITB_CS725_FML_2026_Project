#!/usr/bin/env bash
# Shared settings for all experiment scripts. Override from the shell, e.g.
#   LR_SCRATCH=3e-4 IMB=weighted bash scripts/run_main_grid.sh
set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-7}"
WORKERS="${WORKERS:-2}"

# Fill these in AFTER the LR sweep + imbalance ablation (steps 5 and 6).
LR_SCRATCH="${LR_SCRATCH:-3e-4}"
LR_PRE_FULL="${LR_PRE_FULL:-1e-4}"
LR_PRE_PARTIAL="${LR_PRE_PARTIAL:-1e-4}"
LR_PRE_FROZEN="${LR_PRE_FROZEN:-1e-2}"
IMB="${IMB:-none}"

# run <extra train.py args...>  -- skips nothing, just forwards the common flags
run() {
  python src/train.py --epochs "$EPOCHS" --patience "$PATIENCE" --workers "$WORKERS" "$@"
}
