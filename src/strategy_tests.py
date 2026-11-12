"""Paired tests between fine-tuning strategies (frozen / partial / full) at each data size, and the
layer-wise sweep. Same seed = same training subset, so tests are paired across seeds. No new runs.

    python src/strategy_tests.py
Outputs: results/tables/paired_strategies.csv, layerwise_summary.csv
"""
from itertools import combinations
from pathlib import Path
import pandas as pd
from scipy import stats

R = Path("results"); T = R / "tables"; T.mkdir(parents=True, exist_ok=True)
frames = [pd.read_csv(R / f) for f in ("lowdata.csv", "main_grid.csv", "layerwise.csv") if (R / f).exists()]
d = pd.concat(frames, ignore_index=True)
d = d[(d.init == "pretrained") & (d.imbalance == "none")] if "imbalance" in d else d[d.init == "pretrained"]
d = d.drop_duplicates(["mode", "pct", "seed"], keep="last")


def paired(a, b, col="test_bal_acc"):
    rows = []
    for pct in sorted(d.pct.unique()):
        x = d[(d.pct == pct) & (d["mode"] == a)].set_index("seed")[col]
        y = d[(d.pct == pct) & (d["mode"] == b)].set_index("seed")[col]
        s = x.index.intersection(y.index)
        if len(s) < 3:
            continue
        diff = (x[s] - y[s]).values
        t, p = stats.ttest_1samp(diff, 0.0)
        h = stats.t.ppf(0.975, len(s) - 1) * diff.std(ddof=1) / len(s) ** 0.5
        rows.append(dict(a=a, b=b, pct=pct, n_seeds=len(s), mean_diff=diff.mean(), ci_lo=diff.mean() - h,
                         ci_hi=diff.mean() + h, p=p))
    return rows


modes = [m for m in ["frozen", "partial", "partial34", "partial234", "partial1234", "full"] if m in set(d["mode"])]
out = pd.DataFrame([r for a, b in combinations(modes, 2) for r in paired(a, b)])
if len(out):
    out["p_bonferroni"] = (out.p * len(out)).clip(upper=1)
    out["verdict"] = out.apply(lambda r: "a>b" if r.p_bonferroni < .05 and r.mean_diff > 0 else
                               "b>a" if r.p_bonferroni < .05 else "no detectable difference", axis=1)
    out.round(4).to_csv(T / "paired_strategies.csv", index=False)
    print(out.round(3).to_string(index=False))
lw = d[d["mode"].isin(modes)].groupby(["mode", "pct"]).test_bal_acc.agg(["count", "mean", lambda x: x.std(ddof=1)]) \
    .rename(columns={"<lambda_0>": "std"}).round(3)
lw.to_csv(T / "layerwise_summary.csv")
print(lw.unstack("pct").to_string())
