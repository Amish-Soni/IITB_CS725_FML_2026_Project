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

## Main grid: scratch vs pretrained (full fine-tune), IMB=none, 3 seeds
Test balanced accuracy (mean): 1%: 0.826 vs 0.539, 2%: 0.816 vs 0.599, 5%: 0.860 vs 0.818,
10%: 0.852 vs 0.873, 25%: 0.846 vs 0.849, 50%: 0.876 vs 0.889, 100%: 0.894 vs 0.879
(pretrained vs scratch). 
Pre-training gives a large, clearly significant advantage at 1-2% data
(gap +0.29 / +0.22; scratch near chance at 1%, very unstable at 2% with std 0.17). At 5% the
gap (+0.04) is within seed noise. From 5-10% upwards the two are statistically tied (scratch is
slightly ahead at 10%, -0.02). 
Crossover: between 2% and 5% of the training data, i.e. between 89 and 221 training images
(training pool = 4,420 images: 1,129 NORMAL / 3,291 PNEUMONIA; subsets keep the 1:3 ratio).
The 1% subset has only 44 images (11 NORMAL, 33 PNEUMONIA), which explains why scratch is near
chance there.; caveat: 3 seeds only. 
AUROC is higher for pretrained at every
size (e.g. 0.933 vs 0.673 at 1%, 0.985 vs 0.978 at 100%), so the ranking quality advantage
persists even where balanced accuracy at threshold 0.5 is tied. 
Recall of NORMAL stays at 0.66-0.79 for pretrained at all sizes (see imbalance notes). 
Wall-clock times are noisy (depend on early-stopping epoch and GPU load) and are not used for the cost comparison.

## Low-data deep dive (1/2/5/10%, 3 seeds, IMB=none, 30 epochs, patience 7)
Test balanced accuracy (frozen / partial / full / scratch):
1%: 0.763 / 0.827 / 0.825 / 0.539    2%: 0.787 / 0.838 / 0.816 / 0.599
5%: 0.817 / 0.839 / 0.860 / 0.818    10%: 0.831 / 0.877 / 0.852 / 0.873
- Any pretrained strategy beats scratch by 0.2-0.3 at 1-2%; scratch catches up by 5-10%.
- Partial (last block + head) is statistically tied with full fine-tuning at every size
  (small edge at 10%: 0.877 +/- 0.010 vs 0.852 +/- 0.007) with 8.4M vs 11.2M trainable params.
- Frozen backbone (1,026 trainable params) is the lowest pretrained strategy at every size; clearly
  worse than partial at 1-2% (-0.06 / -0.05), overlapping within noise at 5-10% (std ~0.04).
  Its AUROC (0.91-0.94) is closer to the others than its balanced accuracy.
- Hypothesis that frozen gives near-best accuracy at tiny cost is NOT supported.
- Cost metric: raw wall-clock time (77-160 s everywhere) is dominated by validation (~800 images
  per epoch), data loading and early-stopping epoch, not by compute, and shows no saving for
  frozen. Cost is therefore reported as estimated training time to the best epoch
  (steps x measured sec/step, src/bench_step.py) and trainable parameters.
- Caveats: 3 seeds; std at 1-5% is large (up to 0.08, scratch 2%: 0.14); learning rates were
  tuned at 10% data and reused at all sizes.

## Cost vs performance
Per-step cost measured with src/bench_step.py (batch 32, 224x224, GPU idle): full 104 ms,
partial 44 ms, frozen 31 ms. Estimated training time to best epoch = steps x sec/step.
- Partial fine-tuning is Pareto-better than full fine-tuning at all sizes (same or better accuracy
  at 1/3-1/2 of the cost; at 5% it is ~2.4x cheaper, -0.02 accuracy, within noise).
- At 10% data: partial 0.877 at ~6.1 s vs scratch 0.873 at ~21.7 s vs full 0.852 at ~16.2 s, so
  partial matches scratch accuracy at ~3.5x less training cost.
- Frozen (1,026 params) is cheapest at 1% (0.47 s) but ~6 points less accurate than partial; at
  5-10% it needs more epochs (8.5 s vs 6.1 s at 10%) and stays less accurate, so it is dominated.
- Caveats: estimate excludes patience epochs, data loading and validation; best_epoch is noisy;
  3 seeds; one machine. Raw wall-clock (results CSV train_time_s) is not used.

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