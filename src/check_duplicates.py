"""Duplicate and near-duplicate checks (reviewer feedback: "check for training data duplication").

1. Exact duplicates (identical bytes) within and across train pool / val / test.
2. Near-duplicates: a 256-bit perceptual difference hash. For every val and test image we find the
   nearest image in the training pool. As a baseline we do the same for training-pool images
   against OTHER patients of the pool. If the test (or val) set contained near-copies of training
   images (leakage), its nearest-neighbour distances would be clearly smaller than that baseline.
3. Optional: re-score saved predictions (results/preds, from run_diagnose.sh) after dropping
   duplicated test images, to show that duplicates do not move the test metrics.

    python src/check_duplicates.py
    python src/check_duplicates.py --preds-dir results/preds
Outputs: results/tables/{duplicate_summary,near_duplicates_summary,near_duplicate_pairs,
exact_duplicates,dedup_sensitivity}.csv and results/figures/{near_duplicates,closest_pairs}.png
"""
import argparse
import hashlib
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HASH = 16  # 16x16 difference hash = 256 bits
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def features(path):
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        im.draft("L", (HASH * 8, HASH * 8))  # fast JPEG downscale; no-op for other formats
        g = im.convert("L").resize((HASH + 1, HASH), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return hashlib.md5(raw).hexdigest(), np.packbits((a[:, 1:] > a[:, :-1]).reshape(-1))


def nearest(q, r, qcode=None, rcode=None, chunk=128):
    """Min Hamming distance (and index) from each packed hash in q to the rows of r.
    If qcode/rcode are given, pairs with the same code (same patient) are ignored."""
    d_best = np.empty(len(q), np.int32)
    i_best = np.empty(len(q), np.int64)
    for s in range(0, len(q), chunk):
        x = q[s:s + chunk, None, :] ^ r[None, :, :]
        d = POP[x].sum(axis=2, dtype=np.int32)
        if qcode is not None:
            d[qcode[s:s + chunk, None] == rcode[None, :]] = 10_000
        i_best[s:s + chunk] = d.argmin(axis=1)
        d_best[s:s + chunk] = d.min(axis=1)
    return d_best, i_best


def summarise(name, d):
    return dict(comparison=name, n=len(d), min=int(d.min()), p05=float(np.percentile(d, 5)),
                median=float(np.median(d)), mean=round(float(d.mean()), 1),
                n_le_8=int((d <= 8).sum()), n_le_16=int((d <= 16).sum()), n_le_32=int((d <= 32).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/chest_xray")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--preds-dir", default=None)
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figs", default="results/figures")
    ap.add_argument("--n-pairs", type=int, default=30)
    a = ap.parse_args()
    root, sp, tab, fig_dir = Path(a.data_root), Path(a.splits), Path(a.tables), Path(a.figs)
    tab.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sets = {k: pd.read_csv(sp / f"{k}.csv") for k in ("train_pool", "val", "test")}
    bits = {}
    for k, df in sets.items():
        md5, b = [], []
        for rel in df["path"]:
            m, h = features(root / rel)
            md5.append(m)
            b.append(h)
        df["md5"] = md5
        bits[k] = np.stack(b)
        print(f"hashed {k}: {len(df)} images")

    # ---------------- 1. exact duplicates
    rows = []
    for k, df in sets.items():
        grp = df.groupby("md5").size()
        rows.append(dict(scope=f"within {k}", n_images_affected=int(df.md5.duplicated(keep=False).sum()),
                         n_groups=int((grp > 1).sum())))
    for x, y in [("train_pool", "val"), ("train_pool", "test"), ("val", "test")]:
        shared = set(sets[x].md5) & set(sets[y].md5)
        rows.append(dict(scope=f"{x} & {y}", n_images_affected=int(sets[y].md5.isin(shared).sum()),
                         n_groups=len(shared)))
    dup = pd.DataFrame(rows)
    dup.to_csv(tab / "duplicate_summary.csv", index=False)
    print("\nExact duplicates (identical bytes):")
    print(dup.to_string(index=False))
    allx = pd.concat([df.assign(set=k) for k, df in sets.items()], ignore_index=True)
    allx[allx.md5.duplicated(keep=False)].sort_values("md5")[["md5", "set", "path", "label", "patient"]] \
        .to_csv(tab / "exact_duplicates.csv", index=False)

    # ---------------- 2. near duplicates
    pool, val, test = sets["train_pool"], sets["val"], sets["test"]
    pc = pd.factorize(pool.patient)[0]
    d_val, i_val = nearest(bits["val"], bits["train_pool"])
    d_test, i_test = nearest(bits["test"], bits["train_pool"])
    d_pool, _ = nearest(bits["train_pool"], bits["train_pool"], pc, pc)
    summ = pd.DataFrame([summarise("val -> train pool", d_val), summarise("test -> train pool", d_test),
                         summarise("train pool -> other patients of train pool", d_pool)])
    summ.to_csv(tab / "near_duplicates_summary.csv", index=False)
    print("\nNearest-neighbour Hamming distance of the 256-bit difference hash (0 = identical image;"
          " unrelated chest X-rays are typically far apart):")
    print(summ.to_string(index=False))
    p_tv = stats.mannwhitneyu(d_test, d_val, alternative="less").pvalue
    p_tp = stats.mannwhitneyu(d_test, d_pool, alternative="less").pvalue
    print(f"\nMann-Whitney, are test->pool distances SMALLER than val->pool? p = {p_tv:.3g}")
    print(f"Mann-Whitney, are test->pool distances SMALLER than the pool->pool baseline? p = {p_tp:.3g}")
    print("Reading: leakage would show up as many small distances for test (or val) that the baseline "
          "does not have, and as small p-values here.")

    pairs = []
    for name, df, d, idx in [("test", test, d_test, i_test), ("val", val, d_val, i_val)]:
        for q in np.argsort(d)[:a.n_pairs]:
            pairs.append(dict(query_set=name, query_path=df.path[q], ref_path=pool.path[idx[q]],
                              distance=int(d[q]), same_label=bool(df.label[q] == pool.label[idx[q]]),
                              query_patient=df.patient[q], ref_patient=pool.patient[idx[q]]))
    pd.DataFrame(pairs).to_csv(tab / "near_duplicate_pairs.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.arange(0, 140, 4)
    for d, lab, col in [(d_pool, "train pool -> other patients (baseline)", "#8a8a85"),
                        (d_val, "val -> train pool", "#1baf7a"), (d_test, "test -> train pool", "#2a78d6")]:
        ax.hist(d, bins=bins, density=True, histtype="step", lw=2, color=col, label=lab)
    ax.axvline(16, color="#bbbbbb", ls="--", lw=1)
    ax.set_xlabel("distance to nearest training image (bits of 256)")
    ax.set_ylabel("density")
    ax.set_title("Near-duplicate search", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "near_duplicates.png", dpi=200)

    k = min(6, len(test))
    order = np.argsort(d_test)[:k]
    fig, axes = plt.subplots(k, 2, figsize=(5, 2.4 * k), squeeze=False)
    for r, q in enumerate(order):
        for c, (rel, title) in enumerate([(test.path[q], "test"), (pool.path[i_test[q]], f"train pool (d={d_test[q]})")]):
            with Image.open(root / rel) as im:
                axes[r, c].imshow(im.convert("L"), cmap="gray")
            axes[r, c].set_title(title, fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "closest_pairs.png", dpi=150)
    print(f"\nLook at {fig_dir / 'closest_pairs.png'}: the closest test/train pairs. Genuine near-copies "
          "would look like the same image.")

    # ---------------- 3. dedup sensitivity of saved predictions
    if a.preds_dir and Path(a.preds_dir).exists():
        drop = set(test.loc[test.md5.duplicated() | test.md5.isin(set(pool.md5)), "path"])
        rows = []
        for f in sorted(Path(a.preds_dir).glob("*.csv")):
            p = pd.read_csv(f)
            t = p[p.split == "test"]
            t2 = t[~t.path.isin(drop)]
            if t2.label.nunique() < 2:
                continue
            rows.append(dict(run=f.stem, n_test=len(t), n_test_dedup=len(t2),
                             bal_acc=balanced_accuracy_score(t.label, t.prob >= 0.5),
                             bal_acc_dedup=balanced_accuracy_score(t2.label, t2.prob >= 0.5),
                             auroc=roc_auc_score(t.label, t.prob), auroc_dedup=roc_auc_score(t2.label, t2.prob)))
        if rows:
            s = pd.DataFrame(rows)
            s["delta_bal_acc"] = s.bal_acc_dedup - s.bal_acc
            s["delta_auroc"] = s.auroc_dedup - s.auroc
            s.round(4).to_csv(tab / "dedup_sensitivity.csv", index=False)
            print(f"\nDropping {len(drop)} duplicated test images changes balanced accuracy by "
                  f"{s.delta_bal_acc.mean():+.4f} on average (range {s.delta_bal_acc.min():+.4f} to "
                  f"{s.delta_bal_acc.max():+.4f}) and AUROC by {s.delta_auroc.mean():+.4f}.")
    else:
        print("\n(no --preds-dir given: skipped the dedup sensitivity check)")


if __name__ == "__main__":
    main()
