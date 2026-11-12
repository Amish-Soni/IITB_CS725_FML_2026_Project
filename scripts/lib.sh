#!/usr/bin/env bash
# Helpers for restart-safe experiment loops (sourced after common.sh).
# done_in <csv> <init> <mode> <pct> <seed> : exit 0 if that run is already in the CSV (restart-safe loops)
done_in() {
  [ -f "$1" ] && python - "$@" <<'PYEOF'
import sys, pandas as pd
f, init, mode, pct, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
d = pd.read_csv(f)
sys.exit(0 if ((d.init == init) & (d["mode"] == mode) & (d.pct == pct) & (d.seed == seed)).any() else 1)
PYEOF
}
