# Design decisions log
Working notes for the CS725 project "The Data Starvation Experiment" (chest X-ray pneumonia,
ResNet-18). Everything here feeds the "Methodology", "Results" and "Limitations" sections of the
final report. All numbers come from `results/*.csv` and `results/tables/*.csv`.

## Headline findings (5 seeds at every size; equal-step control, layer-wise sweep, diagnostics done)
1. Under the standard fixed-epoch recipe, pre-training helps clearly only at 1-2% of the training
   pool (44-89 images): +0.20 / +0.26 balanced accuracy. From 5% (221 images) upward we find no
   detectable difference between scratch and pre-trained fine-tuning (5 seeds at all sizes).
2. **The equal-step control shrinks the low-data gap a lot.** When every epoch has the same number of
   optimizer steps (30), scratch at 1% rises from 0.607 to 0.835 and at 2% from 0.560 to 0.814, so
   the near-chance scratch results were mostly under-training under the fixed-epoch recipe, not only
   data scarcity. With equal steps the pre-trained advantage is +0.01 (1%), +0.03 (2%, p = 0.034,
   unadjusted), +0.02 (5%), +0.02 (10%) in balanced accuracy; only 2% has a CI excluding 0. In AUROC
   the pre-trained model stays ahead at every size (0.950 vs 0.913 at 1%, 0.972 vs 0.950 at 10%).
   So: the headline "pre-training rescues tiny datasets" must be stated as "under a fixed-epoch
   budget"; with matched steps the benefit is small in balanced accuracy but persists in ranking
   quality (AUROC) and in sample/step efficiency (pre-trained gets there in fewer steps).
3. Pre-training makes training more reliable under the fixed-epoch recipe (seed std 0.02-0.04 vs
   0.07-0.13 at 1-10%). With equal steps the std is closer (0.018-0.037 vs 0.021-0.048).
4. Fine-tuning strategies: no detectable difference between partial and full, nor between any of
   the layer-wise variants (last 1, 2, 3 or 4 blocks), at any size. A frozen backbone is lowest
   (0.04-0.12 below partial at 1-10%) but with 60 pairwise comparisons nothing survives Bonferroni.
   Partial is 1.4-3x cheaper than full by estimated training cost.
5. Class-imbalance mitigation (weighted loss, oversampling) gave no consistent gain.
6. Validation-to-test gap (0.10-0.13 balanced accuracy for pre-trained models) is mostly a
   threshold/score shift on NORMAL test images, not a loss of ranking ability (see "Diagnosing the
   validation-to-test gap").
7. Reviewer checks: no pretraining overlap by chronology; no exact duplicates across splits; no
   excess of near-duplicates between test/validation and train (see "Reviewer feedback").

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
- Equal-step control: every epoch is exactly 30 optimizer steps (sampling with replacement),
  up to 50 epochs, patience 10 epochs (= 300 steps); same LRs.
- Layer-wise modes: `partial34` (layer3+4), `partial234`, `partial1234` (all blocks, new head, BN
  of frozen parts in eval); `partial` = layer4 only.
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
- [x] Equal-step (equal-budget) control for scratch vs pre-trained at 1-10% (5 seeds)
- [x] Paired statistics for the fine-tuning strategies (`src/strategy_tests.py`)
- [x] Layer-wise freezing sweep (partial34 / partial234 / partial1234), 5 seeds at 1-10%
- [x] 5 seeds at 25-100%
- [x] Duplicate / near-duplicate checks (`src/check_duplicates.py`), pretraining-overlap argument
- [x] Validation-to-test gap diagnosis (`src/diagnose_shift.py`)
- [ ] Final report, final slides (Nov 24-26)
- Decided not to do: second architecture, second dataset (listed under limitations)

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

## Main grid: scratch vs pre-trained (full fine-tune), IMB=none, fixed-epoch recipe
5 seeds (0-4) at every size. Test balanced accuracy, mean (sample std, ddof=1), AUROC in brackets.
Source: `results/tables/main_grid_summary.csv`.

| Data | Images | Pre-trained | Scratch |
|---|---|---|---|
| 1% | 44 | 0.809 (0.039) [0.928] | 0.607 (0.097) [0.712] |
| 2% | 89 | 0.819 (0.018) [0.945] | 0.560 (0.130) [0.751] |
| 5% | 221 | 0.847 (0.025) [0.956] | 0.833 (0.070) [0.923] |
| 10% | 442 | 0.868 (0.023) [0.969] | 0.816 (0.080) [0.943] |
| 25% | 1,105 | 0.860 (0.027) [0.976] | 0.855 (0.035) [0.962] |
| 50% | 2,210 | 0.865 (0.021) [0.981] | 0.885 (0.021) [0.967] |
| 100% | 4,420 | 0.879 (0.026) [0.985] | 0.870 (0.037) [0.978] |

- Clear pre-training advantage only at 1-2% (+0.20 / +0.26). From 5% up the gap is within noise.
  Crossover between 2% and 5% under this recipe (but see the equal-step control below).
- The 3-seed results at 25-100% (pre-trained 0.846/0.876/0.894) moved by up to 0.03 when seeds 3-4
  were added: 3 seeds were not enough; the old numbers are superseded.
- AUROC is higher for pre-trained at every size (gap 0.007-0.22), largest at 1-2%.
- Scratch at 1-2% is near chance (AUROC 0.71 / 0.75): several seeds do not learn at all
  (balanced accuracy 0.50-0.56). The equal-step control shows this is mostly under-training.

### Paired comparison (pre-trained minus scratch; same seed = same training subset)
Source: `results/tables/paired_pretrained_vs_scratch.csv` (regenerate it for 25-100% with 5 seeds;
the file you have still lists n = 3 there). Values below were recomputed from `results/main_grid.csv`.

| Data | n seeds | Mean diff | 95% CI | p |
|---|---|---|---|---|
| 1% | 5 | +0.202 | [+0.05, +0.36] | 0.022 |
| 2% | 5 | +0.258 | [+0.11, +0.41] | 0.008 |
| 5% | 5 | +0.014 | [-0.09, +0.12] | 0.73 |
| 10% | 5 | +0.052 | [-0.08, +0.18] | 0.32 |
| 25% | 5 | +0.005 | [-0.02, +0.03] | 0.65 |
| 50% | 5 | -0.020 | [-0.05, +0.01] | 0.10 |
| 100% | 5 | +0.009 | [-0.04, +0.06] | 0.64 |

- Detectable at 1-2% only; 2% (p = 0.008) only just misses Bonferroni (0.007), so p-values are
  descriptive. At >= 5% use "no detectable difference" (absence of evidence, not equality).

## Equal-step control (scratch vs pre-trained, 30 steps per epoch, 5 seeds)
Source: `results/equal_steps.csv`. Test balanced accuracy, mean (std) [AUROC]:

| Data | Pre-trained | Scratch | Diff | 95% CI | p |
|---|---|---|---|---|---|
| 1% | 0.843 (0.018) [0.950] | 0.835 (0.021) [0.913] | +0.007 | [-0.04, +0.05] | 0.68 |
| 2% | 0.844 (0.033) [0.954] | 0.814 (0.048) [0.920] | +0.030 | [+0.00, +0.06] | 0.034 |
| 5% | 0.861 (0.018) [0.961] | 0.842 (0.041) [0.934] | +0.018 | [-0.03, +0.07] | 0.34 |
| 10% | 0.856 (0.037) [0.972] | 0.837 (0.040) [0.950] | +0.019 | [-0.03, +0.07] | 0.38 |

- Scratch at 1% goes 0.607 -> 0.835 and at 2% 0.560 -> 0.814 when it gets enough optimizer steps;
  all 5 seeds learn (min 0.772 at 2%, vs 0.500 under the fixed-epoch recipe).
- The pre-trained model also improves at 1% (0.809 -> 0.843), so the recipe change helps both.
- Steps to best epoch (mean): pre-trained 78 / 180 / 264 / 324 vs scratch 390 / 360 / 282 / 432
  at 1 / 2 / 5 / 10%. Pre-training reaches its best validation score in far fewer steps at 1-2%
  (about 5x at 1%), roughly equal at 5%, and fewer at 10%.
- Estimated cost (steps x 0.106 s): pre-trained 8 / 19 / 28 / 34 s vs scratch 41 / 38 / 30 / 46 s.
- Interpretation for the report: the large low-data gap in the main grid is mostly an artefact of
  the fixed-epoch budget; the robust, budget-independent advantages of pre-training are (i) better
  ranking quality (AUROC +0.02-0.04), (ii) fewer steps to convergence at 1-2%, (iii) lower variance
  under the fixed-epoch budget. Only one budget (30 steps/epoch, <= 1,500 steps) was tried, so a
  still longer budget could narrow the remaining gap further.

## Low-data strategies (1/2/5/10%, 5 seeds, 30 epochs, patience 7)
Test balanced accuracy, mean +/- sample std (source: `results/tables/lowdata_summary.csv`):

| Data | Frozen | Partial | Full | Scratch |
|---|---|---|---|---|
| 1% | 0.711 +/- 0.121 | 0.835 +/- 0.063 | 0.809 +/- 0.039 | 0.607 +/- 0.097 |
| 2% | 0.782 +/- 0.018 | 0.828 +/- 0.050 | 0.819 +/- 0.018 | 0.560 +/- 0.130 |
| 5% | 0.826 +/- 0.039 | 0.847 +/- 0.046 | 0.847 +/- 0.025 | 0.833 +/- 0.070 |
| 10% | 0.827 +/- 0.042 | 0.871 +/- 0.018 | 0.868 +/- 0.023 | 0.816 +/- 0.080 |

- Paired tests (`results/tables/paired_strategies.csv`): partial vs full: differences -0.00 to +0.03,
  all p > 0.29, no detectable difference. Frozen vs full: -0.10 / -0.04 / -0.02 / -0.04 at 1/2/5/10%
  (frozen vs partial -0.12 / -0.05 / -0.02 / -0.04); the unadjusted p-values are 0.03-0.12 and none
  survive Bonferroni over the 60 comparisons (smallest adjusted p = 0.82), so say "frozen tends to be lower".
- Frozen (1,026 trainable params) is the lowest pre-trained strategy: much worse and unstable at 1%.
  Hypothesis "frozen gives near-best accuracy at tiny cost" is not supported.
- Recall of NORMAL stays 0.58-0.82 for all pre-trained strategies.

### Layer-wise freezing sweep (5 seeds; trainable blocks counted from the end)
Source: `results/layerwise.csv`, `results/tables/layerwise_summary.csv`. Test balanced accuracy:

| Trainable | Params | 1% | 2% | 5% | 10% |
|---|---|---|---|---|---|
| head only (frozen) | 1,026 | 0.711 | 0.782 | 0.826 | 0.827 |
| layer4 (partial) | 8.39M | 0.835 | 0.828 | 0.847 | 0.871 |
| layer3-4 | 10.49M | 0.787 | 0.849 | 0.846 | 0.861 |
| layer2-4 | 11.02M | 0.826 | 0.841 | 0.838 | 0.869 |
| layer1-4 | 11.17M | 0.818 | 0.840 | 0.825 | 0.873 |
| everything (full) | 11.18M | 0.809 | 0.819 | 0.847 | 0.868 |

- Once at least layer4 is trained, accuracy is flat across the sweep (differences <= 0.03, all
  p > 0.14, no detectable difference); the only visible step is from head-only to layer4.
  Training later layers is enough; training earlier layers adds cost, not accuracy.
- Unlike the other tables this sweep has a seed std of 0.02-0.08, so differences of a few points
  cannot be resolved with 5 seeds.

## Cost vs performance
Per-step cost (`src/bench_step.py`, batch 32, 224x224): full 106 ms, layer1-4 92, layer2-4 73,
layer3-4 58, layer4 (partial) 46, frozen 32. Estimated training time to best epoch
= steps to best epoch x sec/step (seconds; fixed-epoch recipe; `cost_vs_performance.csv`):

| Data | Frozen | Partial | Full | Scratch |
|---|---|---|---|---|
| 1% | 0.40 | 0.88 | 2.18 | 0.49 |
| 2% | 1.27 | 1.84 | 2.55 | 0.85 |
| 5% | 4.57 | 3.17 | 8.40 | 7.89 |
| 10% | 7.31 | 5.09 | 15.7 | 20.4 |

- Partial is Pareto-better than full at all sizes (no detectable accuracy difference, 1.4-3x cheaper).
- At 10%: partial 0.871 at ~5.1 s vs scratch 0.816 at ~20 s: ~4x cheaper with equal or better
  accuracy (the accuracy difference is within noise).
- The fixed-epoch table favours scratch at 1-2% only because scratch stopped early at near-chance
  accuracy; under equal steps (see above) scratch needs 5x more steps than pre-trained at 1%.
- Raw wall-clock (`train_time_s`) is not used: it is dominated by per-epoch validation (812
  images), data loading, early-stopping epochs and machine load (one frozen-5% run took thousands
  of seconds, probably a machine suspend).
- Caveats: the estimate excludes patience epochs, data loading and validation; `best_epoch` is
  noisy; single machine; LRs tuned at 10% only. Layer-wise costs: layer3-4 reaches 3.9 s at 10%,
  the cheapest; but the sweep's step counts are noisy and not separately tabulated.

## Diagnosing the validation-to-test gap
Source: `src/diagnose_shift.py`, `results/tables/diagnose_*.csv`, `results/figures/diagnose_scores.png`
(9 reruns with saved probabilities, seeds 0-2). Validation balanced accuracy is 0.94-0.99, test
0.81-0.89 (gap 0.08-0.14 for pre-trained at all sizes; ~0.1 for scratch at >= 5%).
- Pneumonia recall is the same on val and test (0.99). The drop is entirely NORMAL recall:
  pre-trained 10%: 0.956 (val) -> 0.698 (test); pre-trained 100%: 0.988 -> 0.748; scratch 100%:
  0.982 -> 0.809.
- Mean predicted P(pneumonia) for NORMAL images: 0.057 (val) vs 0.320 (test) at pre-trained 10%;
  0.021 vs 0.265 at 100%. The model finds the test NORMAL images more pneumonia-like.
- Ranking is only mildly affected: AUROC 0.994 -> 0.963 (10%), 1.000 -> 0.986 (100%). The best
  possible threshold on test (oracle, diagnostic only) would give 0.905 / 0.943 instead of 0.845 /
  0.871, at a threshold of ~0.97 / ~0.98. So most of the gap is a score shift at the decision
  threshold, not a failure to separate the classes.
- A threshold chosen on validation does not help (test balanced accuracy 0.837 / 0.859, i.e. no
  better than 0.5), since validation scores do not reflect the shifted test scores.
- By image group (pre-trained 100%, recall val -> test): NORMAL `IM-*` files 0.993 -> 0.841,
  NORMAL `NORMAL2-IM-*` files 0.984 -> 0.709 (165 of the 234 test NORMAL images); pneumonia
  bacteria 0.996 -> 0.993, virus 0.989 -> 0.998. Both NORMAL naming schemes drop, the NORMAL2
  group most. Scratch at 100% is better on NORMAL (0.937 / 0.756).
- Simple image statistics (size, brightness, contrast, sample of up to 200/group) do not show a
  large difference between test and train for any group (brightness 0.455-0.509, contrast 0.20-0.24);
  the shift is not explained by those.
- Part of the gap is selection bias: validation balanced accuracy is the maximum over epochs
  on that same set.
- Conclusion for the report: the official test set contains NORMAL images that look systematically
  different to the models than the (patient-split) validation NORMAL images; reasons were not
  identified (acquisition, labelling, or patient mix are possible). Absolute test numbers are
  therefore pessimistic for NORMAL recall, and comparisons are between models on the same fixed test set.
  We did not tune thresholds on test (that would be leakage).

## Reviewer feedback (mid-project)
**Review 1: "Was the model already pretrained on the data used? Unfair for scratch?"**
- The pre-trained model uses torchvision `IMAGENET1K_V1` weights, i.e. ImageNet-1k (ILSVRC-2012),
  natural photographs. The Kermany chest X-ray dataset was published in 2018 (Mendeley Data v3,
  1 June 2018; Cell 2018), years after ImageNet-1k was assembled, so it cannot be part of the
  pretraining data. Pretraining on unrelated natural images and fine-tuning on medical images is the
  standard transfer-learning setting this project studies, not an unfair advantage; the cost of the
  pretraining data is exactly what the "data starvation" question is about.
- Limitation: we cannot check ImageNet images directly offline, so the argument rests on
  chronology plus the empirical duplicate checks below. A second, unseen dataset was considered
  and not used (the reviewer's request was conditional on overlap).
- The fairness concern that does apply is the training budget (not the data): see the equal-step
  control, which gives scratch the same optimizer steps.

**Review 2: "Check training data duplication for the PT model."**
Source: `src/check_duplicates.py`, `results/tables/duplicate_summary.csv`, `near_duplicates_summary.csv`,
`near_duplicate_pairs.csv`, `dedup_sensitivity.csv`, figures `near_duplicates.png`, `closest_pairs.png`.
- Exact duplicates (identical bytes): within pool 36 images in 17 groups (19 extra copies), within val
  14 images / 7 groups, within test 12 / 6; **none across** pool-val, pool-test, val-test.
- Near-duplicates: 256-bit difference hash, distance to the nearest training image. Median 34
  (val), 32 (test), 34 (pool vs other patients of the pool); minimum 15 for val and test (11 in the
  pool). Pairs within 16 bits: val 3/812 (0.4%), test 2/624 (0.3%), pool baseline 22/4420 (0.5%).
  The test and val distances are not smaller than the within-pool baseline, so there is no excess of near
  copies. The six closest test/train pairs were inspected visually: all are different patients
  with similar framing, not the same image.
- Dropping the 6 duplicated test images (extra copies) changes balanced accuracy by -0.002 to +0.003
  and AUROC by at most 0.0005 across the 9 saved runs: duplicates do not move the results.
- The 19 duplicate copies inside the training pool (0.4% of images) were kept; at this size they
  only act as slightly higher sample weights. We did not retrain on a de-duplicated pool.
- Perceptual hashing detects near copies of the same image only; it does not rule out the same
  patient under a different id (see Leakage checks).

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
1. One dataset (chest X-ray) and one architecture (ResNet-18): the crossover size, the size of the
   pre-training effect and the layer-wise findings may not transfer. We deliberately did not add a
   second dataset or architecture.
2. Seeds: 5 everywhere, but confidence intervals at 1-10% are still wide (+/- 0.07-0.18 on paired
   differences).
3. Training budget: under the fixed-epoch recipe small subsets get very few optimizer steps (1 / 2 / 6 /
   13 / 34 / 69 / 138 per epoch at 1 / 2 / 5 / 10 / 25 / 50 / 100%). The equal-step control (30
   steps per epoch, up to 1,500 steps) shows the main-grid scratch collapse at 1-2% is largely
   under-training. Only one equal-step budget was tested; the "fixed-epoch" curve is reported as the
   recipe-dependent result, the equal-step table as the budget-matched one.
4. Learning rates tuned on the 10% subset with seed 0 under the fixed-epoch recipe and reused for
   the equal-step runs and all sizes.
5. Large validation set (812 images) even for a 44-image training subset.
6. The official test set appears harder than validation for NORMAL images (recall 0.70-0.81 vs
   0.96-0.99); explained as a score shift, cause not identified; no threshold tuning on test.
7. Patient identity is not provided by the dataset (ids parsed from file names); within-split duplicates
   (pool 19, val 7, test 6 extra copies) kept.
8. Cost metric is an estimate (see "Cost vs performance").
9. Multiple comparisons: p-values are descriptive; the strategy and layer-wise comparisons (60 tests)
   have no result that survives Bonferroni.
10. Pretraining overlap with ImageNet cannot be verified directly; supported by chronology only.

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
PCTS="25 50 100" SEEDS="3 4" IMB=none bash scripts/run_main_grid.sh     # 5 seeds at 25-100%
bash scripts/run_equal_steps.sh           # equal-step control
bash scripts/run_layerwise.sh             # layer-wise sweep
bash scripts/run_diagnose.sh              # saves predictions for diagnostics
python src/bench_step.py
python src/check_duplicates.py --preds-dir results/preds
python src/diagnose_shift.py
python src/strategy_tests.py
python src/analyze.py all                 # tables in results/tables, figures in results/figures
```
Learning rates are set in `scripts/common.sh` (3e-4 scratch, 1e-4 pre-trained full and partial, 1e-2 frozen). Paired statistics: snippet in the project chat /
`results/tables/paired_pretrained_vs_scratch.csv`.