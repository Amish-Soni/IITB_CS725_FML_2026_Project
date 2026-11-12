# Design decisions log
Working notes for the CS725 project "The Data Starvation Experiment" (chest X-ray pneumonia,
ResNet-18). Everything here feeds the "Methodology", "Results" and "Limitations" sections of the
final report. All numbers come from `results/*.csv` and `results/tables/*.csv`.

## Headline findings (as of 5 seeds at 1-10%, 3 seeds at 25-100%)
1. Pre-training helps clearly only in the very-low-data region (1-2% of the training pool,
   44-89 images). From 5% (221 images) upward we find no detectable difference between scratch and
   pre-trained fine-tuning in balanced accuracy. Crossover: between 2% and 5%.
2. Pre-training also makes training reliable: seed std at 1-10% is 0.02-0.04 (pre-trained) vs
   0.07-0.13 (scratch).
3. Partial fine-tuning (last ResNet block + head) matches full fine-tuning at 1.4-3x lower
   estimated training cost; a frozen backbone is cheapest at 1-2% but clearly less accurate.
4. Class-imbalance mitigation (weighted loss, oversampling) gave no consistent gain.
5. Caveat that matters: small subsets get few optimizer steps under our fixed-epoch recipe (see
   "Known limitations", item 3). An equal-step control is still to be run.

## Splits
- Test set = official Kermany test set (624 images: 234 NORMAL / 390 PNEUMONIA), fixed, never used
  for tuning or model selection.
- Official val (16 images) is too small, so official train + val are pooled (5,232 images) and a
  new validation set (15%, 812 images: 220 NORMAL / 592 PNEUMONIA) is split **by patient** to avoid
  leakage. Remaining training pool: 4,420 images (1,129 NORMAL / 3,291 PNEUMONIA, about 1:3).
- Training subsets (1, 2, 5, 10, 25, 50, 100% of the training pool) are **nested** (the 5% subset
  is contained in the 10% subset, etc.) and **class-stratified**, built for seeds 0, 1, 2, 3, 4
  (`python src/prepare_splits.py --seeds 0 1 2 3 4`; seeds 0-2 are byte-identical to the first run).
- Subset sizes: 1% = 44 (11 N / 33 P), 2% = 89, 5% = 221, 10% = 442, 25% = 1,105, 50% = 2,210,
  100% = 4,420 images. Source: `splits/summary.csv`.
- Limitation: the validation set stays 812 images for every subset. In a real low-data setting
  you would not have such a large validation set, so low-data numbers are somewhat optimistic.

## Training recipe (same for every model)
- ResNet-18 (torchvision, ImageNet1K_V1 weights for pre-trained), new 2-way linear head.
- AdamW, weight decay 1e-4, batch size 32 (`drop_last` when the subset has more than 32 images),
  up to 30 epochs, early stopping on validation balanced accuracy (patience 7); the best-validation
  checkpoint is evaluated once on the test set.
- Inputs: 224x224, grayscale replicated to 3 channels, ImageNet normalisation. Training
  augmentation: RandomResizedCrop(scale 0.8-1.0) + rotation up to 10 degrees, no horizontal flip
  (heart position carries meaning).
- Modes: `full` = all layers trainable (11.18M params); `partial` = layer4 + head (8.39M);
  `frozen` = head only (1,026). BatchNorm layers inside frozen blocks stay in eval mode so ImageNet
  statistics are not overwritten by tiny subsets.
- Same epoch cap and patience for the main grid and the low-data runs (30 / 7), which keeps the
  comparisons like-for-like and matches the budget under which the LRs were tuned.
- Per-condition learning rates: see "Learning rates".

## Metrics
Balanced accuracy (main), macro-F1, AUROC, recall of NORMAL and PNEUMONIA, all at threshold 0.5
except AUROC. Plain accuracy is stored but not used because of the 1:3 class imbalance.

## Experiment checklist
- [x] Data downloaded, patient-aware nested splits generated and committed (seeds 0-4)
- [x] Leakage checks (`src/check_overlap.py`, `src/check_patient_ids.py`)
- [x] LR sweep per condition on validation
- [x] Imbalance ablation (none / weighted / oversample)
- [x] Scratch vs pre-trained (full), all subsets (5 seeds at 1-10%, 3 seeds at 25-100%)
- [x] Paired statistics scratch vs pre-trained
- [x] Frozen vs partial vs full at 1, 2, 5, 10% (5 seeds)
- [x] Cost vs performance (measured per-step cost x steps to best epoch)
- [x] Data-starvation curve and crossover analysis
- [ ] Equal-step (equal-budget) control for scratch vs pre-trained at 1-10%
- [ ] Paired statistics for the fine-tuning strategies (partial vs full vs frozen vs scratch)
- [ ] Optional: layer-wise freezing sweep, second architecture, second dataset, more seeds at 25-100%
- [ ] Final report, mid-project slides (Oct 26), final slides (Nov 24-26)

## Learning rates
Chosen on validation balanced accuracy, 10% subset, seed 0, imbalance=none, then reused for all
sizes: scratch 3e-4, pre-trained full 1e-4, pre-trained partial 1e-4, pre-trained frozen 1e-2.
Grid 3e-5 ... 1e-2 (+ 3e-2, 1e-1 for frozen, because its best value sat on the edge of the first
grid; it then peaked at 1e-2, mid-grid). Limitation: single seed and a single data size, so
differences under about 0.01 are within noise, and the LR may be suboptimal at other sizes.

## Imbalance handling (ablation: none / weighted loss / oversampling)
2 models x {5%, 25%} x 3 seeds, LRs from the sweep. Pre-trained: no detectable difference between
the three methods (test balanced accuracy within 0.006, much smaller than the seed std). Scratch:
inconsistent (weighted best at 5%, oversample worst at 5% but fine at 25%, weighted worst at 25%).
Mean validation balanced accuracy: none 0.960, oversample 0.958, weighted 0.956. No method gave a
reliable gain, so the default is IMB=none (simplest, top validation score); balanced accuracy,
macro-F1, AUROC and per-class recall are reported everywhere.
Observation: test recall of NORMAL stays ~0.72-0.76 for every method while validation balanced
accuracy is ~0.96, which suggests a train/test distribution shift in the official split rather
than a class-prior effect (not verified).

## Main grid: scratch vs pre-trained (full fine-tune), IMB=none
Seeds: 5 at 1/2/5/10% (seeds 0-4), 3 at 25/50/100% (seeds 0-2). Test balanced accuracy, mean
(std over seeds), pre-trained vs scratch:

| Data | Images | Pre-trained | Scratch |
|---|---|---|---|
| 1% | 44 | 0.809 (0.039) | 0.607 (0.097) |
| 2% | 89 | 0.819 (0.018) | 0.560 (0.130) |
| 5% | 221 | 0.847 (0.025) | 0.833 (0.070) |
| 10% | 442 | 0.869 (0.023) | 0.816 (0.080) |
| 25% | 1,105 | 0.846 (0.025) | 0.849 (0.037) |
| 50% | 2,210 | 0.876 (0.018) | 0.889 (0.027) |
| 100% | 4,420 | 0.894 (0.022) | 0.879 (0.024) |

- Clear pre-training advantage only at 1-2% (gap +0.20 / +0.26). From 5% up the gap is within seed
  noise. Crossover: between 2% and 5% of the training pool (89-221 images).
- The scratch advantage at 10% seen with 3 seeds (-0.02) disappeared with 5 seeds (+0.05, within
  noise): 3 seeds were not enough at small sizes.
- Pre-training also reduces variance (see headline finding 2).
- AUROC is higher for pre-trained at every size (0.928 vs 0.712 at 1%, 0.969 vs 0.943 at 10%,
  0.985 vs 0.978 at 100%), so ranking quality favours pre-training even where balanced accuracy at
  threshold 0.5 is tied. AUROC std over seeds not yet computed.
- Scratch at 1-2% is near chance (macro-F1 0.52 / 0.42, AUROC 0.71 / 0.75): some seeds do not
  learn anything. See "Known limitations", item 3, for why this may partly reflect the training
  budget rather than data alone.

### Paired comparison (pre-trained minus scratch; same seed = same training subset)
Source: `results/tables/paired_pretrained_vs_scratch.csv`. Paired t-test across seeds; 95% CI
computed from mean and SE with the t distribution (rounded; the CSV is authoritative).

| Data | n seeds | Mean diff | 95% CI | p |
|---|---|---|---|---|
| 1% | 5 | +0.203 | [+0.05, +0.36] | 0.022 |
| 2% | 5 | +0.258 | [+0.11, +0.41] | 0.008 |
| 5% | 5 | +0.014 | [-0.09, +0.12] | 0.73 |
| 10% | 5 | +0.052 | [-0.08, +0.18] | 0.32 |
| 25% | 3 | -0.003 | [-0.04, +0.03] | 0.69 |
| 50% | 3 | -0.013 | [-0.07, +0.04] | 0.41 |
| 100% | 3 | +0.015 | [-0.09, +0.12] | 0.62 |

- The pre-training advantage is statistically detectable at 1-2% only. With 7 comparisons, 2%
  (p = 0.008) only just misses a Bonferroni threshold (0.007), so p-values are treated as descriptive.
- At >= 5% we find **no detectable difference**. This is absence of evidence, not proof of
  equality: the 95% CI at 5-10% (n = 5) spans roughly -0.09 to +0.18, so differences of several
  points cannot be excluded. Use "no detectable difference", not "tied" or "equal".
- Crossover: between 2% and 5% of the pool (89-221 images); not resolvable more finely with the
  sizes tested.

## Low-data strategies (1/2/5/10%, 5 seeds, 30 epochs, patience 7)
Test balanced accuracy, mean +/- std (frozen / partial / full / scratch):

| Data | Frozen | Partial | Full | Scratch |
|---|---|---|---|---|
| 1% | 0.711 +/- 0.109 | 0.835 +/- 0.056 | 0.809 +/- 0.035 | 0.607 +/- 0.086 |
| 2% | 0.782 +/- 0.016 | 0.828 +/- 0.045 | 0.819 +/- 0.016 | 0.560 +/- 0.116 |
| 5% | 0.826 +/- 0.035 | 0.847 +/- 0.042 | 0.847 +/- 0.023 | 0.833 +/- 0.063 |
| 10% | 0.827 +/- 0.038 | 0.871 +/- 0.016 | 0.868 +/- 0.021 | 0.816 +/- 0.071 |

- Partial (last block + head) shows no detectable difference from full fine-tuning at any size
  (descriptive; not formally tested yet). Its seed std is larger at 1-5% (0.04-0.06 vs 0.02-0.035).
- Frozen (1,026 trainable params) is the lowest pre-trained strategy: much worse and unstable at
  1%, within noise of the others at 5-10% but 0.02-0.04 lower on average. Its AUROC (0.89-0.94)
  is closer to the others than its balanced accuracy.
- Hypothesis "frozen gives near-best accuracy at tiny cost" is not supported.
- Recall of NORMAL stays 0.58-0.82 for all pre-trained strategies.

## Cost vs performance
Per-step cost (`src/bench_step.py`, batch 32, 224x224, GPU idle): full 104 ms, partial 44 ms,
frozen 31 ms. Estimated training time to best epoch = steps to best epoch x sec/step (seconds):

| Data | Frozen | Partial | Full | Scratch |
|---|---|---|---|---|
| 1% | 0.39 | 0.86 | 2.14 | 0.48 |
| 2% | 1.22 | 1.79 | 2.50 | 0.83 |
| 5% | 4.40 | 3.08 | 8.24 | 7.74 |
| 10% | 7.03 | 4.95 | 15.4 | 20.0 |

- Partial is Pareto-better than full at all sizes (no detectable accuracy difference, 1.4-3x cheaper).
- At 10%: partial 0.871 at ~4.9 s vs scratch 0.816 at ~20 s: ~4x cheaper with equal or better
  accuracy (the accuracy difference is within noise).
- Frozen is cheapest at 1-2% but 0.04-0.12 less accurate than partial; at 5-10% it needs more steps
  (4.4 s / 7.0 s) and is still less accurate, so it is dominated.
- Raw wall-clock (`train_time_s`) is not used: it is dominated by per-epoch validation (812
  images), data loading, the early-stopping epoch and machine load (one frozen-5% run took thousands
  of seconds, probably a machine suspend).
- Caveats: the estimate excludes patience epochs, data loading and validation; `best_epoch` is
  noisy; single machine; LRs tuned at 10% only.

## Leakage checks (`src/check_overlap.py`, `src/check_patient_ids.py`)
- Train pool vs val: no shared patient ids, file names or identical images (split by patient).
- Train pool vs official test: 0 byte-identical images, 0 identical file names. 144 pneumonia and
  81 normal ids are shared by *number* only:
  - Pneumonia: the infection type in the file name never matches (0/144, vs 43% expected by
    chance) and image numbers differ by hundreds (median gap 315). The dataset numbers bacterial
    and viral patients separately, so `person1_bacteria_*` and `person1_virus_*` are different people.
  - Normal: 0/81 shared ids have the same filename prefix; the pool copy is `IM-xxxx-*` while the
    test copy is `NORMAL2-IM-xxxx-*`, i.e. two separate numbering schemes reusing the same numbers.
  Conclusion: no evidence of patient overlap between the pool and the official test set.
- Our patient key (`person{N}` / `IM-{N}` with the prefix dropped) merges patients that share a
  number across subtypes/schemes into one group. This is conservative for train/val (groups are
  only larger, so no leakage) but "unique patients" counts (pool 2,252, val 398, test 404) are
  underestimates.
- Exact duplicate images exist within a split (pool 19, val 7, test 6), none across splits.
- Not excluded: the same patient imaged under a different id; the dataset does not provide patient
  identity, so absence of overlap is supported by the checks above but not guaranteed.

## Known limitations (for the report)
1. One dataset (chest X-ray) and one architecture (ResNet-18): the crossover size may not transfer.
2. Seeds: 5 at 1-10%, 3 at 25-100%; confidence intervals are wide, especially at 25-100%.
3. **Fixed-epoch budget:** with batch size 32 and `drop_last`, a subset yields 1 / 2 / 6 / 13 / 34 /
   69 / 138 optimizer steps per epoch at 1 / 2 / 5 / 10 / 25 / 50 / 100%. With at most 30 epochs and
   early stopping, 1% gets at most 30 steps and 2% at most 60. Scratch training needs many more
   steps than that, so the near-chance scratch results at 1-2% may partly reflect under-training
   and not only data scarcity; the crossover size could be overstated. An equal-step control is
   planned and should be reported next to the main curve.
4. Learning rates tuned on the 10% subset with seed 0 and reused at all sizes.
5. Large validation set (812 images) even for a 44-image training subset.
6. Official test set appears harder than validation (balanced accuracy ~0.96 on validation vs
   ~0.85-0.89 on test; NORMAL recall 0.66-0.79): a distribution shift not yet diagnosed. No
   threshold tuning; all thresholded metrics use 0.5.
7. Patient identity is not provided by the dataset (ids parsed from file names); within-split duplicates.
8. Cost metric is an estimate (see "Cost vs performance").
9. Multiple comparisons: p-values are descriptive.

## Reproducing the results
```bash
pip install -r requirements.txt
python src/prepare_splits.py --seeds 0 1 2 3 4
python src/check_overlap.py; python src/check_patient_ids.py
bash scripts/run_lr_sweep.sh              # then frozen extension: lr 3e-2, 1e-1 (see Learning rates)
bash scripts/run_imbalance.sh
IMB=none bash scripts/run_main_grid.sh                                  # seeds 0-2, all sizes
PCTS="1 2 5 10" SEEDS="3 4" IMB=none bash scripts/run_main_grid.sh      # extra seeds
LOWDATA_EPOCHS=30 LOWDATA_PATIENCE=7 IMB=none bash scripts/run_lowdata.sh
LOWDATA_EPOCHS=30 LOWDATA_PATIENCE=7 SEEDS="3 4" IMB=none bash scripts/run_lowdata.sh
python src/bench_step.py
python src/analyze.py all                 # tables in results/tables, figures in results/figures
```
Learning rates are set in `scripts/common.sh`. Paired statistics: snippet in the project chat /
`results/tables/paired_pretrained_vs_scratch.csv`.