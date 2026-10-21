"""Micro-benchmark: seconds per optimiser step for each fine-tuning mode.

Why: wall-clock time in results.csv is dominated by things that are NOT the cost of the
strategy (JPEG decoding, per-epoch validation on ~800 images, early-stopping epoch, GPU load).
This measures the pure forward+backward+update cost of one batch, per mode, on a fixed
random batch. analyze.py then estimates training cost = steps-to-best-epoch x sec_per_step.

Run once, with the GPU otherwise idle:   python src/bench_step.py
Output: results/step_time.csv
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from models import build_model, count_trainable


def bench(mode, batch, size, device, warmup=10, steps=50):
    model = build_model("pretrained", mode).to(device)
    # same BN handling as train.py: BN of frozen blocks stays in eval mode
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, size, size, device=device)
    y = torch.randint(0, 2, (batch,), device=device)

    def step():
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps, count_trainable(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="results/step_time.csv")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for mode in ["full", "partial", "frozen"]:
        sec, n = bench(mode, args.batch_size, args.img_size, device, steps=args.steps)
        rows.append(dict(mode=mode, sec_per_step=sec, trainable_params=n, device=str(device)))
        print(f"{mode:8s} {sec * 1000:7.1f} ms/step   trainable={n:,}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
