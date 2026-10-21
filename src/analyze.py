"""Turn results/*.csv into tables (results/tables) and plots (results/figures).

python src/analyze.py lr          # step 5: which LR per condition (validation only)
python src/analyze.py imbalance   # step 6: none vs weighted vs oversample
python src/analyze.py grid        # step 7: data-starvation curve + crossover
python src/analyze.py lowdata     # step 8: frozen / partial / full / scratch at 1-10%
python src/analyze.py cost        # step 8: cost vs performance
python src/analyze.py all
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results")
TAB, FIG = R / "tables", R / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# fixed colour per condition (never reassigned): blue, orange, aqua, magenta
COLOR = {"scratch/full": "#eb6834", "pretrained/full": "#2a78d6",
         "pretrained/partial": "#1baf7a", "pretrained/frozen": "#e87ba4"}
LABEL = {"scratch/full": "Scratch (full)", "pretrained/full": "Pretrained, full fine-tune",
         "pretrained/partial": "Pretrained, last block", "pretrained/frozen": "Pretrained, frozen"}
METRIC = "test_bal_acc"


def load(name):
    f = R / f"{name}.csv"
    if not f.exists():
        sys.exit(f"missing {f} - run the matching scripts/run_*.sh first")
    d = pd.read_csv(f)
    d["cond"] = d["init"] + "/" + d["mode"]
    return d


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def ms(x):
    return f"{x.mean():.3f} ± {x.std(ddof=0):.3f}"


# ---------------------------------------------------------------- step 5
def lr():
    d = load("lr_sweep")
    t = d.pivot_table(index="lr", columns="cond", values="val_bal_acc").sort_index(ascending=False)
    t.to_csv(TAB / "lr_sweep_val.csv")
    print("Validation balanced accuracy by learning rate (higher is better):\n")
    print(t.round(3).to_string())
    print("\nBest LR per condition (by VALIDATION):")
    for c in t.columns:
        print(f"  {c:20s} -> {t[c].idxmax():g}   (val_bal_acc={t[c].max():.3f})")
    print("\nTip: if two LRs are within ~0.01, prefer the smaller (more stable) one.")


# ---------------------------------------------------------------- step 6
def imbalance():
    d = load("imbalance")
    g = d.groupby(["init", "pct", "imbalance"])
    t = g.agg(val_bal_acc=("val_bal_acc", "mean"),
              test_bal_acc=(METRIC, ms), macro_f1=("test_macro_f1", ms),
              auroc=("test_auroc", ms), recall_normal=("test_recall_normal", ms),
              recall_pneumonia=("test_recall_pneumonia", ms)).round(3)
    t.to_csv(TAB / "imbalance.csv")
    print(t.to_string())
    best = d.groupby("imbalance")["val_bal_acc"].mean().sort_values(ascending=False)
    print("\nMean VALIDATION balanced accuracy across all runs (use this to choose):")
    print(best.round(3).to_string())
    print(f"\n-> suggested default: IMB={best.index[0]}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, init in zip(axes, ["scratch", "pretrained"]):
        s = d[d["init"] == init]
        m = s.groupby(["pct", "imbalance"])[METRIC].agg(["mean", "std"]).reset_index()
        pcts = sorted(m.pct.unique())
        w = 0.25
        for i, (imb, col) in enumerate(zip(["none", "weighted", "oversample"],
                                           ["#8a8a85", "#2a78d6", "#eb6834"])):
            r = m[m.imbalance == imb].set_index("pct").reindex(pcts)
            ax.bar(np.arange(len(pcts)) + (i - 1) * w, r["mean"], w * 0.9, yerr=r["std"],
                   color=col, label=imb, capsize=2)
        ax.set_xticks(range(len(pcts))); ax.set_xticklabels([f"{p}%" for p in pcts])
        style(ax, "training data", "test balanced accuracy", init)
        ax.set_ylim(0.4, 1.0)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "imbalance.png", dpi=200)


# ---------------------------------------------------------------- step 7
def grid():
    d = load("main_grid")
    t = d.groupby(["cond", "pct"]).agg(
        n_runs=("seed", "count"), bal_acc=(METRIC, "mean"), bal_acc_std=(METRIC, "std"),
        macro_f1=("test_macro_f1", "mean"), auroc=("test_auroc", "mean"),
        recall_normal=("test_recall_normal", "mean"), time_s=("train_time_s", "mean")).round(4)
    t.to_csv(TAB / "main_grid_summary.csv")
    print(t.to_string())

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c in ["scratch/full", "pretrained/full"]:
        s = t.loc[c].reset_index()
        ax.errorbar(s.pct, s.bal_acc, yerr=s.bal_acc_std.fillna(0), marker="o", ms=6,
                    lw=2, capsize=3, color=COLOR[c], label=LABEL[c])
    ax.set_xscale("log"); ax.set_xticks(sorted(d.pct.unique()))
    ax.set_xticklabels([f"{p}%" for p in sorted(d.pct.unique())])
    ax.legend(frameon=False)
    style(ax, "training data used (log scale)", "test balanced accuracy (mean ± std over seeds)",
          "Data-starvation curve")
    fig.tight_layout(); fig.savefig(FIG / "data_starvation_curve.png", dpi=200)

    print("\nCrossover analysis (gap = pretrained - scratch):")
    print("  'within noise' means |gap| <= max(0.01, combined std across seeds)")
    a, b = t.loc["pretrained/full"], t.loc["scratch/full"]
    rows = []
    for pct in a.index:
        gap = a.loc[pct, "bal_acc"] - b.loc[pct, "bal_acc"]
        noise = float(np.hypot(a.loc[pct, "bal_acc_std"], b.loc[pct, "bal_acc_std"]))
        rows.append(dict(pct=pct, gap=round(gap, 4), noise=round(noise, 4),
                         within_noise=abs(gap) <= max(0.01, noise)))
    c = pd.DataFrame(rows)
    c.to_csv(TAB / "crossover.csv", index=False)
    print(c.to_string(index=False))
    ok = c[c.within_noise]
    print("\n-> pre-training stops mattering from about:",
          f"{int(ok.pct.min())}% of the data" if len(ok) else "no size in your grid (still helps at 100%)")


# ---------------------------------------------------------------- step 8
def _lowdata_frame():
    parts = [load("main_grid"), load("lowdata")]
    d = pd.concat(parts, ignore_index=True)
    return d[d.pct <= 10]


def lowdata():
    d = _lowdata_frame()
    t = d.groupby(["cond", "pct"]).agg(
        n_runs=("seed", "count"), bal_acc=(METRIC, ms), macro_f1=("test_macro_f1", ms),
        auroc=("test_auroc", ms), recall_normal=("test_recall_normal", ms),
        trainable_params=("trainable_params", "first"), time_s=("train_time_s", "mean")).round(3)
    t.to_csv(TAB / "lowdata_summary.csv")
    print(t.to_string())

    m = d.groupby(["cond", "pct"])[METRIC].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c in COLOR:
        s = m[m.cond == c]
        if len(s):
            ax.errorbar(s.pct, s["mean"], yerr=s["std"].fillna(0), marker="o", ms=6, lw=2,
                        capsize=3, color=COLOR[c], label=LABEL[c])
    pcts = sorted(d.pct.unique())
    ax.set_xscale("log"); ax.set_xticks(pcts); ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.legend(frameon=False, fontsize=8)
    style(ax, "training data used (log scale)", "test balanced accuracy", "Low-data regime")
    fig.tight_layout(); fig.savefig(FIG / "lowdata_strategies.png", dpi=200)


def cost():
    """Cost vs performance. Cost = estimated training time to the selected checkpoint
    = (optimiser steps up to best epoch) x (measured seconds per step for that mode).
    Needs results/step_time.csv from `python src/bench_step.py`. Raw wall-clock is also kept
    in the table but is NOT used for the plot: it is dominated by data loading, per-epoch
    validation (~800 images) and GPU load, not by the strategy's compute."""
    d = _lowdata_frame()
    st = R / "step_time.csv"
    if not st.exists():
        sys.exit("missing results/step_time.csv - run: python src/bench_step.py")
    sps = pd.read_csv(st).set_index("mode")["sec_per_step"]
    # scratch/full has the same per-step cost as pretrained/full
    d["steps_to_best"] = d["best_epoch"] * np.maximum(1, d["n_train"] // 32)
    d["est_time"] = d["steps_to_best"] * d["mode"].map(sps)
    m = d.groupby(["cond", "pct"]).agg(
        bal=(METRIC, "mean"), auroc=("test_auroc", "mean"), est_time=("est_time", "mean"),
        wall_time=("train_time_s", "mean"), params=("trainable_params", "first")).reset_index()
    m.to_csv(TAB / "cost_vs_performance.csv", index=False)
    pcts = sorted(m.pct.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, xl in [(axes[0], "est_time", "est. training time to best epoch (s, log scale)"),
                        (axes[1], "params", "trainable parameters (log scale)")]:
        for c in COLOR:
            s = m[m.cond == c].sort_values("pct")
            if not len(s):
                continue
            ax.plot(s[key], s.bal, color=COLOR[c], lw=1, alpha=0.5)
            ax.scatter(s[key], s.bal, s=20 + 12 * np.log2(s.pct + 1), color=COLOR[c],
                       edgecolor="white", linewidth=1.5, label=LABEL[c], zorder=3)
        ax.set_xscale("log")
        style(ax, xl, "test balanced accuracy", "")
    axes[0].set_title("Cost vs performance (bigger dot = more data: "
                      f"{pcts[0]}-{pcts[-1]}%)", loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "cost_vs_performance.png", dpi=200)
    print(m.round(3).to_string(index=False))

if __name__ == "__main__":
    cmds = dict(lr=lr, imbalance=imbalance, grid=grid, lowdata=lowdata, cost=cost)
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "all":
        for f in cmds.values():
            f()
    elif arg in cmds:
        cmds[arg]()
    else:
        sys.exit(__doc__)
