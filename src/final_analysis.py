"""Final-report analysis: paired scratch-vs-pretrained tests (main grid + equal-step control) and
the two new figures. Reads results/*.csv, writes results/tables/ and results/figures/.

    python src/final_analysis.py
Outputs:
  results/tables/paired_pretrained_vs_scratch.csv   (overwritten: 5 seeds at every size)
  results/tables/paired_equal_steps.csv
  results/tables/equal_steps_summary.csv
  results/tables/layerwise_table.csv
  results/figures/equal_steps_vs_fixed.png
  results/figures/layerwise.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

R = Path("results"); T = R / "tables"; F = R / "figures"
T.mkdir(parents=True, exist_ok=True); F.mkdir(parents=True, exist_ok=True)
COL = {"pretrained": "#2a78d6", "scratch": "#e07b39"}


def paired(df, a="pretrained", b="scratch", col="test_bal_acc"):
    rows = []
    for pct in sorted(df.pct.unique()):
        x = df[(df.init == a) & (df.pct == pct)].drop_duplicates("seed", keep="last").set_index("seed")[col]
        y = df[(df.init == b) & (df.pct == pct)].drop_duplicates("seed", keep="last").set_index("seed")[col]
        s = x.index.intersection(y.index)
        d = (x[s] - y[s]).dropna()
        if len(d) < 3:
            continue
        se = d.std(ddof=1) / len(d) ** 0.5
        h = stats.t.ppf(0.975, len(d) - 1) * se
        rows.append(dict(pct=pct, n_seeds=len(d), mean_diff=d.mean(), se=se, ci95_lo=d.mean() - h,
                         ci95_hi=d.mean() + h, p_value=stats.ttest_1samp(d, 0.0).pvalue))
    return pd.DataFrame(rows)


def agg(df):
    g = df.groupby(["init", "pct"]).agg(n=("seed", "nunique"), bal=("test_bal_acc", "mean"),
                                         sd=("test_bal_acc", lambda x: x.std(ddof=1)),
                                         auroc=("test_auroc", "mean")).reset_index()
    return g


# ---------- 1. main grid paired test (5 seeds)
m = pd.read_csv(R / "main_grid.csv")
if "imbalance" in m:
    m = m[m.imbalance == "none"]
m = m[m["mode"] == "full"]
p1 = paired(m)
p1.round(4).to_csv(T / "paired_pretrained_vs_scratch.csv", index=False)
print("Paired pretrained - scratch, main grid (fixed-epoch recipe):")
print(p1.round(3).to_string(index=False))

# ---------- 2. equal-step control
e = pd.read_csv(R / "equal_steps.csv")
p2 = paired(e)
p2.round(4).to_csv(T / "paired_equal_steps.csv", index=False)
print("\nPaired pretrained - scratch, equal-step control:")
print(p2.round(3).to_string(index=False))
ge = agg(e)
ge.round(4).to_csv(T / "equal_steps_summary.csv", index=False)

gm = agg(m)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, g) in zip(axes, [("Fixed-epoch recipe (main grid)", gm[gm.pct <= 10]),
                                ("Equal-step control (30 steps / epoch)", ge)]):
    for init in ("pretrained", "scratch"):
        s = g[g.init == init].sort_values("pct")
        ax.errorbar(range(len(s)), s.bal, yerr=s.sd, marker="o", capsize=3, lw=2, color=COL[init], label=init)
    ax.set_xticks(range(4)); ax.set_xticklabels(["1%", "2%", "5%", "10%"])
    ax.set_title(name, loc="left", fontsize=10); ax.set_xlabel("training data")
    ax.grid(axis="y", alpha=.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("test balanced accuracy (mean ± std, 5 seeds)")
axes[0].legend(frameon=False)
axes[0].set_ylim(0.45, 0.95)
fig.tight_layout(); fig.savefig(F / "equal_steps_vs_fixed.png", dpi=200)

# ---------- 3. layer-wise
frames = [pd.read_csv(R / f) for f in ("lowdata.csv", "main_grid.csv", "layerwise.csv") if (R / f).exists()]
d = pd.concat(frames, ignore_index=True)
if "imbalance" in d:
    d = d[d.imbalance == "none"]
d = d[(d.init == "pretrained") & (d.pct <= 10)].drop_duplicates(["mode", "pct", "seed"], keep="last")
order = [("frozen", "head only"), ("partial", "layer4"), ("partial34", "layer3-4"),
         ("partial234", "layer2-4"), ("partial1234", "layer1-4"), ("full", "everything")]
tab = []
for mode, label in order:
    row = {"trainable": label, "params": int(d[d["mode"] == mode].trainable_params.iloc[0])}
    for pct in (1, 2, 5, 10):
        x = d[(d["mode"] == mode) & (d.pct == pct)].test_bal_acc
        row[f"{pct}%"] = f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"
    tab.append(row)
tab = pd.DataFrame(tab)
tab.to_csv(T / "layerwise_table.csv", index=False)
print("\nLayer-wise table:")
print(tab.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 4))
cm = plt.cm.Blues
for k, pct in enumerate((1, 2, 5, 10)):
    ys = [d[(d["mode"] == mo) & (d.pct == pct)].test_bal_acc.mean() for mo, _ in order]
    ax.plot(range(len(order)), ys, marker="o", lw=2, color=cm(0.35 + 0.18 * k), label=f"{pct}% data")
ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], rotation=20, fontsize=8)
ax.set_ylabel("test balanced accuracy (mean, 5 seeds)"); ax.set_xlabel("trainable part of ResNet-18")
ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(F / "layerwise.png", dpi=200)