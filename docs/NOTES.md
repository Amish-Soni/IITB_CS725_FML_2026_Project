# Design decisions log

Keep adding to this as you go; it becomes the "Methodology" and "Limitations" sections.

## Splits
- Test set = official Kermany test set, fixed, never used for tuning.
- Official val (16 images) is too small, so train + val are pooled and a new validation set
  (15%) is split **by patient** to avoid leakage.
- Training subsets (1, 2, 5, 10, 25, 50, 100% of the training pool) are **nested** and
  **class-stratified**, built for seeds 0, 1, 2.
- Limitation: the validation set stays the same size for every subset. In a real low-data
  setting you would not have such a large validation set, so low-data numbers are slightly
  optimistic.
- Subset sizes (identical for seeds 0-2): 1% = 44, 2% = 89, 5% = 221, 10% = 442, 25% = 1,105,
  50% = 2,210, 100% = 4,420 images. Source: splits/summary.csv.

## Training recipe (same for every model)
- ResNet-18, AdamW, weight decay 1e-4, batch size 32, up to 30 epochs.
- Early stopping on validation balanced accuracy (patience 7); test evaluated once at the best epoch.
- Inputs: 224x224, grayscale replicated to 3 channels, ImageNet normalisation, light augmentation.
- Learning rate differs per condition and is tuned on validation only.

## Metrics
Balanced accuracy (main), macro-F1, AUROC, recall of NORMAL and PNEUMONIA. Plain accuracy is
reported but not trusted because of class imbalance.

## Experiment checklist
- [ ] Phase 1: data downloaded, splits generated and committed
- [ ] Phase 2: LR sweep per condition on validation
- [ ] Phase 2: imbalance ablation (none / weighted / oversample)
- [ ] Phase 2: scratch vs pretrained, all subsets, 3 seeds
- [ ] Phase 4: frozen vs partial vs full at 1, 2, 5, 10%
- [ ] Phase 4: cost vs performance plot (train time, trainable params)
- [ ] Phase 5: data-starvation curve and crossover analysis

## Learning rates
Chosen on validation balanced accuracy, 10% subset, seed 0, imbalance=none, then reused for all
sizes: scratch 3e-4, pretrained-full 1e-4, pretrained-partial 1e-4, pretrained-frozen 1e-2
(swept 3e-5..1e-1; frozen peaked mid-grid). Limitation: single seed, so differences under
about 0.01 are within noise.

## Imbalance handling (ablation: none / weighted loss / oversampling)
2 models x {5%, 25%} x 3 seeds, LRs from the sweep. Pretrained: all three methods tied (test
balanced accuracy within 0.006, much smaller than the seed std). Scratch: inconsistent (weighted
best at 5%, oversample worst at 5% but fine at 25%, weighted worst at 25%). Mean validation
balanced accuracy: none 0.960, oversample 0.958, weighted 0.956. No method gave a reliable gain,
so the default is IMB=none (simplest, top validation score); we still report balanced accuracy,
macro-F1, AUROC and per-class recall everywhere. Observation: test recall of NORMAL stays ~0.72-0.76
for every method while validation balanced accuracy is ~0.96, which suggests a train/test
distribution shift in the official split rather than a class-prior effect (not verified).

## Main grid: scratch vs pretrained (full fine-tune), IMB=none
Seeds: 5 at 1/2/5/10% (seeds 0-4), 3 at 25/50/100% (seeds 0-2). Test balanced accuracy mean,
pretrained vs scratch: 1%: 0.809 vs 0.607, 2%: 0.819 vs 0.560, 5%: 0.847 vs 0.833,
10%: 0.869 vs 0.816, 25%: 0.846 vs 0.849, 50%: 0.876 vs 0.889, 100%: 0.894 vs 0.879.
- Clear pre-training advantage only at 1-2% (gap +0.20 / +0.26, ~4 SE). From 5% up the gap is
  within seed noise. Crossover: between 2% and 5% of the training pool (89-221 images).
- The scratch advantage at 10% seen with 3 seeds (-0.02) disappeared with 5 seeds (+0.05, within
  noise): 3 seeds were not enough at small sizes.
- Pre-training also reduces variance: seed std at 1-10% is 0.02-0.04 (pretrained) vs 0.07-0.13 (scratch).
- AUROC is higher for pretrained at every size (0.928 vs 0.712 at 1%, 0.969 vs 0.943 at 10%,
  0.985 vs 0.978 at 100%).

## Low-data strategies (1/2/5/10%, 5 seeds, 30 epochs, patience 7)
Test balanced accuracy (frozen / partial / full / scratch):
1%: 0.711 / 0.835 / 0.809 / 0.607    2%: 0.782 / 0.828 / 0.819 / 0.560
5%: 0.826 / 0.847 / 0.847 / 0.833    10%: 0.827 / 0.871 / 0.868 / 0.816
- Partial (last block + head) is statistically tied with full fine-tuning at every size.
- Frozen (1,026 trainable params) is the lowest pretrained strategy: much worse and unstable at
  1% (0.711 +/- 0.109), within noise of the others at 5-10% but 0.02-0.04 lower on average.
- Hypothesis "frozen gives near-best accuracy at tiny cost" is not supported.

## Cost vs performance
Per-step cost (src/bench_step.py, batch 32, 224x224): full 104 ms, partial 44 ms, frozen 31 ms.
Estimated training time to best epoch = steps x sec/step:
1%: frozen 0.39 s / partial 0.86 / full 2.14 / scratch 0.48
2%: 1.22 / 1.79 / 2.50 / 0.83      5%: 4.40 / 3.08 / 8.24 / 7.74      10%: 7.03 / 4.95 / 15.4 / 20.0
- Partial is Pareto-better than full at all sizes (same accuracy, 1.4-3x cheaper).
- At 10%: partial 0.871 at ~4.9 s vs scratch 0.816 at ~20 s: ~4x cheaper, equal or better
  accuracy (difference within noise).
- Frozen is cheapest at 1-2% but 0.04-0.12 less accurate than partial; at 5-10% it needs more
  steps (4.4 s / 7.0 s) and is still less accurate, so it is dominated.
- Raw wall-clock (train_time_s) is not used: it is dominated by validation, data loading, early
  stopping and machine load (one run in the frozen-5% cell took thousands of seconds,
  probably a machine suspend).
- Caveats: estimate excludes patience epochs, data loading and validation; best_epoch is noisy;
  one machine; LRs tuned on the 10% subset only.

## Leakage checks (src/check_overlap.py, src/check_patient_ids.py)
- Train pool vs val: no shared patient ids, file names or identical images (split by patient).
- Train pool vs official test: 0 byte-identical images, 0 identical file names. 144 pneumonia and
  81 normal ids are shared by *number* only:
  - Pneumonia: the infection type in the file name never matches (0/144, vs 43% expected by
    chance) and image numbers differ by hundreds (median gap 315). The dataset numbers bacterial
    and viral patients separately, so `person1_bacteria_*` and `person1_virus_*` are different people.
  - Normal: 0/81 shared ids have the same filename prefix; the pool copy is `IM-xxxx-*` while the
    test copy is `NORMAL2-IM-xxxx-*`, i.e. two separate numbering schemes reusing the same numbers.
  Conclusion: no evidence of patient overlap between the pool and the official test set.
- Our patient key (`person{N}` / `IM-{N}` with the prefix dropped) merges patients that share a number
  across subtypes/schemes into one group. This is conservative for train/val (groups are only larger,
  so no leakage) but "unique patients" counts (pool 2,252, val 398, test 404) are underestimates.
- Exact duplicate images exist within a split (pool 19, val 7, test 6), none across splits.
- Not excluded: the same patient imaged under a different id; the dataset does not provide patient
  identity, so absence of overlap is supported by the checks above but not guaranteed.