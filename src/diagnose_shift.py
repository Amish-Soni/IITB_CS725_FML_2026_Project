"""Why is validation balanced accuracy (~0.96) so much higher than test (~0.85-0.89)?

Uses (a) the val/test columns that every results CSV already has, and (b) the saved probabilities
from scripts/run_diagnose.sh (results/preds/*.csv). It looks at:
  1. the val-minus-test gap per condition and data size
  2. ranking quality (AUROC) vs the 0.5 decision threshold: if AUROC transfers but balanced accuracy
     does not, the scores are shifted, not the features
  3. recall by image group (NORMAL file-name scheme IM- vs NORMAL2-IM-, pneumonia bacteria vs virus)
  4. score distributions of val vs test images
  5. simple image statistics (size, brightness, contrast) per split and class

    python src/diagnose_shift.py
Outputs: results/tables/diagnose_*.csv, results/figures/diagnose_scores.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

GRID = np.round(np.arange(0.02, 0.981, 0.02), 2)
FNAME = re.compile(r"^(scratch|pretrained)_(\w+?)_p(\d+)_s(\d+)\.csv$")


def rates(y, p, t=0.5):
    pred = p >= t
    tnr = float((~pred[y == 0]).mean()) if (y == 0).any() else np.nan
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
    return tnr, tpr, (tnr + tpr) / 2


def best_threshold(y, p):
    s = np.array([rates(y, p, t)[2] for t in GRID])
    i = int(np.nanargmax(s))
    return float(GRID[i]), float(s[i])


def group_of(path, label):
    name = Path(path).name
    if label == 0:
        return "NORMAL2-IM-* files" if name.startswith("NORMAL2") else "NORMAL IM-* files"
    m = re.search(r"_(bacteria|virus)_", name)
    return f"PNEUMONIA {m.group(1)}" if m else "PNEUMONIA other"


def gap_from_runs(res, tab):
    frames = []
    for name in ["main_grid", "lowdata", "layerwise", "equal_steps"]:
        f = res / f"{name}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["file"] = name
            frames.append(d)
    if not frames:
        print("1. no results CSVs found, skipped")
        return
    d = pd.concat(frames, ignore_index=True)
    g = d.groupby(["file", "init", "mode", "pct"]).agg(
        n=("seed", "count"), val_bal=("val_bal_acc", "mean"), test_bal=("test_bal_acc", "mean"),
        test_auroc=("test_auroc", "mean")).reset_index()
    g["gap"] = g.val_bal - g.test_bal
    g.round(4).to_csv(tab / "diagnose_gap_from_runs.csv", index=False)
    m = g[g.file == "main_grid"]
    print("1. Validation minus test balanced accuracy, main grid (mean over seeds):")
    if len(m):
        print(m.pivot_table(index=["init", "mode"], columns="pct", values="gap").round(3).to_string())
    print("   Note: val_bal_acc is the BEST epoch on the validation set (selected on it), so it is\n"
          "   optimistically biased by construction; some of the gap is this selection effect.\n")


def load_preds(pred_dir):
    parts = []
    for f in sorted(pred_dir.glob("*.csv")):
        m = FNAME.match(f.name)
        if not m:
            continue
        p = pd.read_csv(f)
        p["init"], p["mode"], p["pct"], p["seed"] = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        parts.append(p)
    return pd.concat(parts, ignore_index=True) if parts else None


def per_run_table(allp):
    rows = []
    for (init, mode, pct, seed), r in allp.groupby(["init", "mode", "pct", "seed"]):
        v, t = r[r.split == "val"], r[r.split == "test"]
        yv, pv, yt, pt = v.label.values, v.prob.values, t.label.values, t.prob.values
        tn_v, tp_v, b_v = rates(yv, pv)
        tn_t, tp_t, b_t = rates(yt, pt)
        thr, _ = best_threshold(yv, pv)
        thr_o, b_o = best_threshold(yt, pt)
        rows.append(dict(init=init, mode=mode, pct=pct, seed=seed,
                         val_auroc=roc_auc_score(yv, pv), test_auroc=roc_auc_score(yt, pt),
                         val_bal=b_v, test_bal=b_t, val_recall_normal=tn_v, test_recall_normal=tn_t,
                         val_recall_pneu=tp_v, test_recall_pneu=tp_t,
                         thr_val_tuned=thr, test_bal_at_val_thr=rates(yt, pt, thr)[2],
                         thr_oracle=thr_o, test_bal_oracle=b_o,
                         mean_p_normal_val=pv[yv == 0].mean(), mean_p_normal_test=pt[yt == 0].mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--data-root", default="data/chest_xray")
    ap.add_argument("--no-image-stats", action="store_true")
    a = ap.parse_args()
    res, tab = Path(a.results), Path(a.results) / "tables"
    figs = Path(a.results) / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    gap_from_runs(res, tab)

    allp = load_preds(res / "preds")
    if allp is None:
        print("No saved predictions in results/preds. Run: bash scripts/run_diagnose.sh")
    else:
        runs = per_run_table(allp)
        runs.round(4).to_csv(tab / "diagnose_runs.csv", index=False)
        cfg = runs.groupby(["init", "mode", "pct"]).agg(n_seeds=("seed", "count"), **{
            c: (c, "mean") for c in ["val_auroc", "test_auroc", "val_bal", "test_bal",
                                      "val_recall_normal", "test_recall_normal", "val_recall_pneu",
                                      "test_recall_pneu", "thr_val_tuned", "test_bal_at_val_thr",
                                      "thr_oracle", "test_bal_oracle", "mean_p_normal_val",
                                      "mean_p_normal_test"]}).round(3)
        cfg.to_csv(tab / "diagnose_summary.csv")
        print("2. Ranking (AUROC) vs decision threshold, mean over seeds:")
        print(cfg[["n_seeds", "val_auroc", "test_auroc", "val_bal", "test_bal", "test_bal_at_val_thr",
                   "test_bal_oracle", "thr_oracle"]].to_string())
        print("   test_bal = balanced accuracy at threshold 0.5; test_bal_at_val_thr = with the threshold that\n"
              "   is best on validation; test_bal_oracle = best possible threshold ON TEST (diagnostic only,\n"
              "   not a legitimate result).\n")
        print("   Reading guide:")
        for (init, mode, pct), r in cfg.iterrows():
            d_auc = r.val_auroc - r.test_auroc
            d_thr = r.test_bal_oracle - r.test_bal
            if d_auc < 0.03 and d_thr > 0.03:
                msg = (f"AUROC drops only {d_auc:.3f} but the best threshold would add {d_thr:+.3f}: the ranking "
                       "transfers, the scores are shifted (calibration/threshold shift)")
            elif d_auc >= 0.03:
                msg = f"AUROC itself drops by {d_auc:.3f}: the learned features separate the test classes worse (feature shift)"
            else:
                msg = "val and test behave similarly"
            if abs(r.test_bal_at_val_thr - r.test_bal) < 0.01 and d_thr > 0.03:
                msg += "; a validation-tuned threshold does not fix it"
            print(f"   {init}/{mode} {pct}%: {msg}")
        print()

        allp["group"] = [group_of(p, l) for p, l in zip(allp.path, allp.label)]
        allp["correct"] = ((allp.prob >= 0.5).astype(int) == allp.label)
        per = allp.groupby(["init", "mode", "pct", "seed", "group", "split"]).agg(
            n=("correct", "size"), recall=("correct", "mean")).reset_index()
        strata = per.groupby(["init", "mode", "pct", "group", "split"]).agg(
            n_images=("n", "mean"), recall=("recall", "mean")).round(3).reset_index()
        strata.to_csv(tab / "diagnose_strata.csv", index=False)
        print("3. Recall at threshold 0.5 by image group (mean over seeds):")
        print(strata.pivot_table(index=["init", "mode", "pct", "group"], columns="split",
                                 values=["n_images", "recall"]).round(3).to_string())
        print()

        cfgs = allp[["init", "mode", "pct"]].drop_duplicates().values.tolist()[:3]
        fig, axes = plt.subplots(len(cfgs), 2, figsize=(9, 2.8 * len(cfgs)), squeeze=False)
        bins = np.linspace(0, 1, 26)
        for r_i, (init, mode, pct) in enumerate(cfgs):
            sub = allp[(allp.init == init) & (allp["mode"] == mode) & (allp.pct == pct)]
            ns = max(sub.seed.nunique(), 1)
            for c_i, (label, name) in enumerate([(0, "NORMAL images"), (1, "PNEUMONIA images")]):
                ax = axes[r_i, c_i]
                for split, col in [("val", "#1baf7a"), ("test", "#2a78d6")]:
                    x = sub[(sub.split == split) & (sub.label == label)].prob.values
                    ax.hist(x, bins=bins, density=True, histtype="step", lw=2, color=col,
                            label=f"{split} (n={len(x) // ns})")
                ax.set_yscale("log")
                ax.set_title(f"{init}/{mode} {pct}%: {name}", loc="left", fontsize=9)
                ax.set_xlabel("predicted P(pneumonia)")
                ax.legend(frameon=False, fontsize=7)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(figs / "diagnose_scores.png", dpi=200)
        print(f"   figure: {figs / 'diagnose_scores.png'}  (if test NORMAL scores sit higher than val NORMAL "
              "scores, test NORMAL images look more pneumonia-like to the model)\n")

    sp, root = Path(a.splits), Path(a.data_root)
    if not a.no_image_stats and root.exists() and (sp / "val.csv").exists():
        rows = []
        for k in ("train_pool", "val", "test"):
            df = pd.read_csv(sp / f"{k}.csv")
            df["grp"] = [group_of(p, l) for p, l in zip(df.path, df.label)]
            for grp, g in df.groupby("grp"):
                for rel in g.path.sample(min(len(g), 200), random_state=0):
                    with Image.open(root / rel) as im:
                        w, h = im.size
                        im.draft("L", (256, 256))
                        x = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32) / 255
                    rows.append(dict(set=k, group=grp, width=w, height=h, brightness=x.mean(), contrast=x.std()))
        st = pd.DataFrame(rows).groupby(["group", "set"]).agg(
            n=("width", "size"), width=("width", "median"), height=("height", "median"),
            brightness=("brightness", "mean"), contrast=("contrast", "mean")).round(3)
        st.to_csv(tab / "diagnose_image_stats.csv")
        print("5. Image statistics (sample of up to 200 images per set and group):")
        print(st.to_string())


if __name__ == "__main__":
    main()
