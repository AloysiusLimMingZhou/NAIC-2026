# Experiment Record: SwinV2-Base + RandomErasing (BEST)

**Date:** 2026-04-08
**Model:** Swin Transformer V2 Base (`microsoft/swinv2-base-patch4-window8-256`)
**Person:** Yi Hui
**Status:** BEST RESULT SO FAR

## Strategy

- 5-Fold Stratified K-Fold (scikit-learn)
- Focal Loss (gamma=1.0, label_smoothing=0.1)
- WeightedRandomSampler
- Data Augmentation: HFlip, VFlip, Rotation(15), ColorJitter(0.2, 0.2, 0.1), RandomErasing(p=0.5)
- Input resolution: 224x224

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Batch size | 16 |
| Epochs | 20 |
| Early stopping | patience=8 |
| Gradient clipping | max_norm=1.0 |

## K-Fold Aggregated Results (5 Folds)

| Metric    | Mean   | Std    |
|-----------|--------|--------|
| Accuracy  | 73.75% | 4.38%  |
| Precision | 66.57% | 4.76%  |
| Recall    | 59.25% | 3.97%  |
| F1 Score  | 58.91% | 4.40%  |
| ROC-AUC   | 87.16% | 3.30%  |

### Per-Fold Validation Results

| Fold | Acc    | Prec   | Rec    | F1     | AUC    | Early Stop |
|------|--------|--------|--------|--------|--------|------------|
| 1    | 74.84% | 60.76% | 55.85% | 56.06% | 84.06% | Epoch 9    |
| 2    | 71.90% | 64.65% | 57.07% | 55.10% | 83.29% | Epoch 10   |
| 3    | 76.71% | 66.27% | 64.07% | 62.22% | 90.16% | Epoch 11   |
| 4    | 66.33% | 65.93% | 55.21% | 55.21% | 86.60% | Epoch 11   |
| 5    | 78.99% | 75.25% | 64.05% | 65.95% | 91.70% | Epoch 12   |

## Test Results (Best model from Fold 5)

| Metric    | Baseline | Ours   | Diff    |
|-----------|----------|--------|---------|
| Accuracy  | 74.19%   | 75.61% | +1.42%  |
| Precision | 69.13%   | 67.29% | -1.84%  |
| Recall    | 58.78%   | 69.80% | +11.02% |
| F1 Score  | 59.95%   | 66.86% | +6.91%  |
| ROC-AUC   | 78.91%   | 94.31% | +15.40% |

## Confusion Matrix

```
[[368  76   3   0   1]
 [  2  86  14   0   3]
 [  2  48 191  10  26]
 [  0   1  17  19   7]
 [  0   2  20   9  83]]
```

## Observations

- Beats baseline on ALL metrics except Precision (-1.84%)
- Massive Recall improvement (+11.02%) — most important metric
- RandomErasing (p=0.5) significantly reduced overfitting compared to previous runs
- Train acc peaked at ~90% vs 93%+ without RandomErasing — better generalization
- Best model came from Fold 5, which had the highest validation scores
- Higher fold std (4.38%) than previous best (1.87%) but much better test performance
