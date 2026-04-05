# Experiment Record: SwinV2 + K-Fold + Focal Loss + Augmentation

**Date:** 2026-04-05
**Model:** Swin Transformer V2 (`microsoft/swinv2-base-patch4-window8-256`)
**Person:** Yi Hui

## Strategy

- 5-Fold Stratified K-Fold (scikit-learn)
- Focal Loss (gamma=2.0, label_smoothing=0.1)
- WeightedRandomSampler (oversample minority classes)
- Data Augmentation: HFlip, VFlip, Rotation(15), ColorJitter(0.2, 0.2, 0.1)
- Input resolution: 224x224

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Batch size | 16 |
| Epochs | 20 |
| Early stopping | patience=5 |
| Gradient clipping | max_norm=1.0 |

## K-Fold Aggregated Results (5 Folds)

| Metric    | Mean   | Std    |
|-----------|--------|--------|
| Accuracy  | 71.98% | 4.83%  |
| Precision | 66.04% | 1.89%  |
| Recall    | 61.71% | 2.81%  |
| F1 Score  | 60.52% | 2.87%  |
| ROC-AUC   | 88.38% | 2.54%  |

### Per-Fold Validation Results

| Fold | Acc    | Prec   | Rec    | F1     | AUC    | Early Stop |
|------|--------|--------|--------|--------|--------|------------|
| 1    | 64.48% | 63.15% | 56.92% | 55.37% | 83.75% | Epoch 8    |
| 2    | 76.96% | 65.68% | 60.68% | 60.70% | 88.19% | Epoch 6    |
| 3    | 73.42% | 68.47% | 65.06% | 64.25% | 90.43% | Epoch 6    |
| 4    | 68.48% | 65.20% | 63.74% | 61.31% | 90.93% | Epoch 6    |
| 5    | 76.58% | 67.71% | 62.15% | 60.99% | 88.58% | Epoch 9    |

## Test Results (Best model from Fold 4)

| Metric    | Baseline | Ours   | Diff    |
|-----------|----------|--------|---------|
| Accuracy  | 74.19%   | 62.96% | -11.23% |
| Precision | 69.13%   | 61.17% | -7.96%  |
| Recall    | 58.78%   | 65.57% | +6.79%  |
| F1 Score  | 59.95%   | 55.13% | -4.82%  |
| ROC-AUC   | 78.91%   | 91.93% | +13.02% |

## Confusion Matrix

```
[[348 100   0   0   0]
 [  2  87   9   5   2]
 [  0  79  91  91  16]
 [  0   2   5  36   1]
 [  0   9  13  32  60]]
```

## Observations

- Recall (primary metric) beat baseline by +6.79%
- ROC-AUC improved significantly (+13.02%), indicating better class discrimination confidence
- Accuracy and precision dropped — model is over-predicting minority classes at cost of majority class (class 0)
- Class 0: 100 samples misclassified as class 1
- Class 2: heavy confusion with class 3 (91 misclassified)

## Next Steps

- Try focal gamma=1.0 (less aggressive) to recover accuracy/precision
- Try increasing epochs to 30
- Try different learning rate (3e-5, 1e-4)
