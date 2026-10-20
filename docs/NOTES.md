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
(pretrained vs scratch). Pre-training gives a large, clearly significant advantage at 1-2% data
(gap +0.29 / +0.22; scratch near chance at 1%, very unstable at 2% with std 0.17). At 5% the
gap (+0.04) is within seed noise. From 5-10% upwards the two are statistically tied (scratch is
slightly ahead at 10%, -0.02). Crossover: between 2% and 5% of the training data, i.e. between 89 and 221 training images
(training pool = 4,420 images: 1,129 NORMAL / 3,291 PNEUMONIA; subsets keep the 1:3 ratio).
The 1% subset has only 44 images (11 NORMAL, 33 PNEUMONIA), which explains why scratch is near
chance there.; caveat: 3 seeds only. AUROC is higher for pretrained at every
size (e.g. 0.933 vs 0.673 at 1%, 0.985 vs 0.978 at 100%), so the ranking quality advantage
persists even where balanced accuracy at threshold 0.5 is tied. Recall of NORMAL stays at
0.66-0.79 for pretrained at all sizes (see imbalance notes). Wall-clock times are noisy (depend on
early-stopping epoch and GPU load) and are not used for the cost comparison.