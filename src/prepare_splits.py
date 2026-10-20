"""Step 1: build leakage-safe, nested, stratified data splits.

What this script does
---------------------
1. Scans data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}.
2. Keeps the OFFICIAL test set as our fixed test set (never touched again).
3. Pools official train + val (the official val has only 16 images, which is useless)
   and carves out our own validation set, split BY PATIENT so the same patient
   never appears in both train and val.
4. From the remaining training pool, builds NESTED subsets
   (1% < 2% < 5% < 10% < 25% < 50% < 100%), stratified by class, for several seeds.
   "Nested" means the 5% subset is fully contained in the 10% subset, and so on,
   so differences between sizes come from data quantity, not sampling luck.

Output: CSV files with columns path,label,cls,patient in ./splits
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ["NORMAL", "PNEUMONIA"]  # label 0 = NORMAL (minority), label 1 = PNEUMONIA
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]


def patient_id(cls: str, name: str) -> str:
    """Parse a patient id from Kermany file names.

    PNEUMONIA: person1_bacteria_1.jpeg -> patient 1
    NORMAL:    IM-0115-0001.jpeg       -> patient 0115
    Falls back to the file name (one 'patient' per image) if the pattern is unknown.
    """
    if cls == "PNEUMONIA":
        m = re.match(r"person(\d+)_", name)
    else:
        m = re.search(r"IM-(\d+)", name)
    return f"{cls}_{m.group(1)}" if m else f"{cls}_{name}"


def scan(root: Path) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        for label, cls in enumerate(CLASSES):
            folder = root / split / cls
            for p in sorted(folder.glob("*.jp*g")):
                rows.append(
                    dict(
                        path=str(p.relative_to(root)),
                        label=label,
                        cls=cls,
                        official_split=split,
                        patient=patient_id(cls, p.name),
                    )
                )
    if not rows:
        raise SystemExit(f"No images found under {root}. Check the dataset path.")
    return pd.DataFrame(rows)


def class_counts(df: pd.DataFrame) -> str:
    c = df["label"].value_counts().to_dict()
    return f"NORMAL={c.get(0, 0)}, PNEUMONIA={c.get(1, 0)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/chest_xray")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42, help="seed for the train/val split")
    args = ap.parse_args()

    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = scan(root)
    test = df[df.official_split == "test"].reset_index(drop=True)
    pool = df[df.official_split != "test"].reset_index(drop=True)

    # patient-level train/val split
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.split_seed)
    tr_idx, va_idx = next(gss.split(pool, pool["label"], groups=pool["patient"]))
    train_pool = pool.iloc[tr_idx].reset_index(drop=True)
    val = pool.iloc[va_idx].reset_index(drop=True)

    leak = set(train_pool.patient) & set(val.patient)
    assert not leak, f"patient leakage between train and val: {len(leak)} patients"
    test_overlap = set(pool.patient) & set(test.patient)

    train_pool.to_csv(out / "train_pool.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    print(f"train pool: {len(train_pool):5d}  ({class_counts(train_pool)})")
    print(f"val       : {len(val):5d}  ({class_counts(val)})")
    print(f"test      : {len(test):5d}  ({class_counts(test)})")
    print(f"patients shared between pool and test: {len(test_overlap)}")

    summary = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        # one random order per class; taking a prefix of it makes subsets nested
        orders = {}
        for label in (0, 1):
            idx = train_pool.index[train_pool.label == label].to_numpy()
            orders[label] = rng.permutation(idx)

        prev = set()
        for frac in FRACTIONS:
            take = []
            for label in (0, 1):
                n = max(2, int(round(frac * len(orders[label]))))
                take.extend(orders[label][:n])
            sub = train_pool.loc[sorted(take)]
            cur = set(sub.index)
            assert prev <= cur, "subsets are not nested!"
            prev = cur

            d = out / f"seed{seed}"
            d.mkdir(exist_ok=True)
            pct = int(round(frac * 100))
            sub.to_csv(d / f"train_{pct:03d}.csv", index=False)
            summary.append(
                dict(seed=seed, pct=pct, n=len(sub),
                     n_normal=int((sub.label == 0).sum()),
                     n_pneumonia=int((sub.label == 1).sum()))
            )

    s = pd.DataFrame(summary)
    s.to_csv(out / "summary.csv", index=False)
    print("\nSubset sizes (seed 0):")
    print(s[s.seed == args.seeds[0]].drop(columns="seed").to_string(index=False))


if __name__ == "__main__":
    main()
