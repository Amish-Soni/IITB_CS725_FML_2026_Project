"""Check for leakage between train pool, val and test: identical images and identical file names."""
import argparse, hashlib, os
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data/chest_xray")
ap.add_argument("--splits", default="splits")
a = ap.parse_args()
root, sp = Path(a.data_root), Path(a.splits)

d = {k: pd.read_csv(sp / f"{k}.csv") for k in ("train_pool", "val", "test")}
for k, df in d.items():
    df["md5"] = df["path"].map(lambda p: hashlib.md5((root / p).read_bytes()).hexdigest())
    df["name"] = df["path"].map(os.path.basename)
    print(f"{k:10s} images={len(df):5d} unique patients={df.patient.nunique():5d} "
          f"unique image hashes={df.md5.nunique():5d}")

def compare(x, y):
    dup_img = set(d[x].md5) & set(d[y].md5)
    dup_name = set(d[x].name) & set(d[y].name)
    shared_pat = set(d[x].patient) & set(d[y].patient)
    print(f"\n{x} vs {y}:")
    print(f"  identical image files (same bytes): {len(dup_img)}")
    print(f"  identical file names              : {len(dup_name)}")
    print(f"  shared patient ids                : {len(shared_pat)}")
    if dup_name:
        # same name but different bytes => numbering reused for different images
        names = sorted(dup_name)[:5]
        same_bytes = 0
        for n in dup_name:
            hx = set(d[x][d[x].name == n].md5); hy = set(d[y][d[y].name == n].md5)
            same_bytes += bool(hx & hy)
        print(f"  of the shared names, same bytes   : {same_bytes} / {len(dup_name)}  (e.g. {names})")
    n_test_aff = d[y].md5.isin(d[x].md5).sum()
    print(f"  {y} images that also exist in {x} : {n_test_aff} ({100 * n_test_aff / len(d[y]):.1f}%)")

compare("train_pool", "val")
compare("train_pool", "test")
compare("val", "test")
