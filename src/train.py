"""Train one configuration and append its test metrics to a results CSV.

Examples
--------
# from scratch on the 5% subset, seed 0
python src/train.py --init scratch --mode full --pct 5 --seed 0

# ImageNet-pretrained, frozen backbone, 2% subset, class-weighted loss
python src/train.py --init pretrained --mode frozen --pct 2 --seed 0 --imbalance weighted

# equal-step control: every epoch is exactly 30 optimiser steps (sampling with replacement),
# whatever the subset size
python src/train.py --init scratch --mode full --pct 1 --seed 0 --steps-per-epoch 30 --epochs 50

# also save validation + test probabilities (for src/diagnose_shift.py)
python src/train.py --init pretrained --mode full --pct 100 --seed 0 --save-preds
"""
import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from data import XrayDataset, get_transforms
from models import MODES, build_model, count_trainable

# Starting-point learning rates (used only when --lr is not given; the scripts always pass --lr).
DEFAULT_LR = {("scratch", "full"): 1e-3, ("pretrained", "full"): 1e-4,
              ("pretrained", "partial"): 3e-4, ("pretrained", "frozen"): 1e-3}
FALLBACK_LR = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_train_mode(model):
    """model.train(), but keep BatchNorm layers of frozen blocks in eval mode so their
    ImageNet running statistics are not overwritten by our tiny dataset."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        ps.append(p.cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def compute_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(
        acc=accuracy_score(y, pred),
        bal_acc=balanced_accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro"),
        auroc=roc_auc_score(y, p),
        recall_normal=recall_score(y, pred, pos_label=0),
        recall_pneumonia=recall_score(y, pred, pos_label=1),
    )


def append_row(out: Path, row: dict):
    """Append one result row. If the CSV already exists, keep ITS columns (extra new columns are
    dropped, missing ones left empty) so older CSVs never get misaligned rows."""
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if out.exists() and out.stat().st_size > 0:
        cols = pd.read_csv(out, nrows=0).columns.tolist()
        dropped = [c for c in df.columns if c not in cols]
        if dropped:
            print(f"  note: {out} has no columns {dropped}; not stored for this file")
        df = df.reindex(columns=cols)
        df.to_csv(out, mode="a", header=False, index=False)
    else:
        df.to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/chest_xray")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--init", choices=["scratch", "pretrained"], required=True)
    ap.add_argument("--mode", choices=list(MODES), default="full")
    ap.add_argument("--pct", type=int, required=True, help="training subset: 1,2,5,10,25,50,100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imbalance", choices=["none", "weighted", "oversample"], default="none")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="0 = one pass over the subset per epoch (default). N > 0 = every epoch is "
                         "exactly N optimiser steps, sampling with replacement (equal-step control)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--tag", default="", help="free-text label, e.g. 'lr-sweep'")
    ap.add_argument("--save-preds", action="store_true",
                    help="save val+test probabilities of the best checkpoint to --pred-dir")
    ap.add_argument("--pred-dir", default="results/preds")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr or DEFAULT_LR.get((args.init, args.mode), FALLBACK_LR)

    # ---- data ----
    tr = XrayDataset(f"{args.splits}/seed{args.seed}/train_{args.pct:03d}.csv", args.data_root,
                     get_transforms(True, args.img_size))
    va = XrayDataset(f"{args.splits}/val.csv", args.data_root, get_transforms(False, args.img_size))
    te = XrayDataset(f"{args.splits}/test.csv", args.data_root, get_transforms(False, args.img_size))

    counts = np.bincount(tr.labels, minlength=2)
    n_samples = args.steps_per_epoch * args.batch_size if args.steps_per_epoch > 0 else len(tr)
    sampler = None
    class_w = None
    if args.imbalance == "oversample":
        w = 1.0 / counts[tr.labels]  # rare class images get drawn more often
        sampler = WeightedRandomSampler(w, num_samples=n_samples, replacement=True)
    elif args.steps_per_epoch > 0:
        sampler = RandomSampler(tr, replacement=True, num_samples=n_samples)
    if args.imbalance == "weighted":
        class_w = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)

    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    tr_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                           shuffle=sampler is None, drop_last=len(tr) > args.batch_size, **kw)
    va_loader = DataLoader(va, batch_size=64, **kw)
    te_loader = DataLoader(te, batch_size=64, **kw)
    steps_per_epoch = len(tr_loader)

    # ---- model ----
    model = build_model(args.init, args.mode).to(device)
    n_trainable = count_trainable(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    print(f"{args.init}/{args.mode} pct={args.pct} seed={args.seed} n_train={len(tr)} "
          f"counts={counts.tolist()} lr={lr} trainable={n_trainable:,} "
          f"steps/epoch={steps_per_epoch} device={device}")

    # ---- train with early stopping on validation balanced accuracy ----
    best, best_state, bad, best_epoch, epochs_run = -1.0, None, 0, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        set_train_mode(model)
        total, seen = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            seen += len(y)
        epochs_run = epoch
        yv, pv = predict(model, va_loader, device)
        val_bal = balanced_accuracy_score(yv, (pv >= 0.5).astype(int))
        print(f"  epoch {epoch:2d}  train_loss={total / max(seen, 1):.4f}  val_bal_acc={val_bal:.4f}")
        if val_bal > best:
            best, best_state, bad, best_epoch = val_bal, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stop")
                break
    train_time = time.time() - t0

    # ---- final test evaluation with the best-validation weights ----
    model.load_state_dict(best_state)
    yt, pt = predict(model, te_loader, device)
    m = compute_metrics(yt, pt)
    row = dict(init=args.init, mode=args.mode, pct=args.pct, seed=args.seed,
               imbalance=args.imbalance, lr=lr, n_train=len(tr), trainable_params=n_trainable,
               best_epoch=best_epoch, val_bal_acc=best, train_time_s=round(train_time, 1),
               tag=args.tag, steps_per_epoch=steps_per_epoch, epochs_run=epochs_run,
               steps_to_best=best_epoch * steps_per_epoch, steps_total=epochs_run * steps_per_epoch,
               **{f"test_{k}": v for k, v in m.items()})
    print("  TEST:", {k: round(v, 4) for k, v in m.items()})

    if args.save_preds:
        yv, pv = predict(model, va_loader, device)  # best checkpoint, not the last epoch
        preds = pd.concat([
            pd.DataFrame({"split": "val", "path": va.df["path"].values, "label": yv, "prob": pv}),
            pd.DataFrame({"split": "test", "path": te.df["path"].values, "label": yt, "prob": pt}),
        ])
        pdir = Path(args.pred_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        preds.to_csv(pdir / f"{args.init}_{args.mode}_p{args.pct:03d}_s{args.seed}.csv", index=False)

    append_row(Path(args.out), row)


if __name__ == "__main__":
    main()
