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