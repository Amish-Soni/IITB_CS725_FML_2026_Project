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
