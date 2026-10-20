# The Data Starvation Experiment

CS725 project: how much target data do we need before ImageNet pre-training stops helping?
Dataset: Chest X-Ray Pneumonia (Kermany et al.), model: ResNet-18.

## Repo layout

```
src/prepare_splits.py   build patient-safe, nested, stratified subsets
src/data.py             dataset + transforms
src/models.py           scratch / pretrained x full / partial / frozen
src/train.py            train one config, append test metrics to results/results.csv
splits/                 generated CSVs (commit these so everyone uses identical splits)
results/results.csv     one row per run (commit this)
docs/NOTES.md           design decisions log (feeds the final report)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

Download "Chest X-Ray Images (Pneumonia)" from Kaggle (paultimothymooney/chest-xray-pneumonia)
and unzip so that the layout is:

```
data/chest_xray/train/{NORMAL,PNEUMONIA}
data/chest_xray/val/{NORMAL,PNEUMONIA}
data/chest_xray/test/{NORMAL,PNEUMONIA}
```

`data/` is git-ignored.

## Run

```bash
python src/prepare_splits.py                                  # once, then commit splits/
python src/train.py --init scratch    --mode full --pct 5 --seed 0
python src/train.py --init pretrained --mode full --pct 5 --seed 0
```

See `docs/NOTES.md` for the experiment plan.
