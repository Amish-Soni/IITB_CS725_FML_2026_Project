"""Are the patient ids shared between train pool and official test the SAME patients?
For pneumonia files the name carries the infection type (bacteria/virus). Same patient => same type."""
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="splits")
a = ap.parse_args()

def load(k):
    df = pd.read_csv(Path(a.splits) / f"{k}.csv")
    df["name"] = df["path"].map(lambda p: Path(p).name)
    df["subtype"] = df["name"].str.extract(r"_(bacteria|virus)_")[0]
    df["num"] = pd.to_numeric(df["name"].str.extract(r"_(\d+)\.jpe?g$")[0], errors="coerce")
    return df

pool, test = load("train_pool"), load("test")
pn, tn = pool[pool.cls == "PNEUMONIA"], test[test.cls == "PNEUMONIA"]
shared = sorted(set(pn.patient) & set(tn.patient))
n_norm = len(set(pool[pool.cls == "NORMAL"].patient) & set(test[test.cls == "NORMAL"].patient))
print(f"shared ids: PNEUMONIA={len(shared)}  NORMAL={n_norm}")

# chance agreement if ids were unrelated: sum_k P_pool(k) * P_test(k) over patient-level subtype
def dist(df):
    return df.groupby("patient")["subtype"].agg(lambda s: s.mode().iat[0]).value_counts(normalize=True)
dp, dt = dist(pn), dist(tn)
chance = sum(dp.get(k, 0) * dt.get(k, 0) for k in set(dp.index) | set(dt.index))

agree, gaps = [], []
for pid in shared:
    x, y = pn[pn.patient == pid], tn[tn.patient == pid]
    agree.append(bool(set(x.subtype.dropna()) & set(y.subtype.dropna())))
    gaps.append(min(abs(i - j) for i in x.num.dropna() for j in y.num.dropna()))
g = pd.Series(gaps)
print(f"\nsubtype agreement among shared pneumonia ids: {100 * sum(agree) / len(agree):.1f}%"
      f"   (expected by chance if unrelated: {100 * chance:.1f}%)")
print(f"min image-number gap between pool and test image of the same id: "
      f"median={g.median():.0f}, 25%={g.quantile(.25):.0f}, 75%={g.quantile(.75):.0f}")
print("\nexamples (pool names | test names):")
for pid in shared[:8]:
    print(f"  {pid:18s} {sorted(pn[pn.patient == pid].name)[:2]} | {sorted(tn[tn.patient == pid].name)[:2]}")
