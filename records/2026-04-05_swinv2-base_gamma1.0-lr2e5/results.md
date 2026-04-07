# Experiment Record: SwinV2-Base + Gamma 1.0 + LR 2e-5 (BEST)

**Date:** 2026-04-05
**Model:** Swin Transformer V2 Base (`microsoft/swinv2-base-patch4-window8-256`)
**Person:** Yi Hui
**Status:** BEST RESULT SO FAR

## Strategy

- 5-Fold Stratified K-Fold (scikit-learn)
- Focal Loss (gamma=1.0, label_smoothing=0.1)
- WeightedRandomSampler
- Data Augmentation: HFlip, VFlip, Rotation(15), ColorJitter(0.2, 0.2, 0.1)
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
| Accuracy  | 77.65% | 1.87%  |
| Precision | 65.16% | 5.62%  |
| Recall    | 58.92% | 1.96%  |
| F1 Score  | 59.77% | 3.20%  |
| ROC-AUC   | 87.63% | 1.84%  |

### Per-Fold Validation Results

| Fold | Acc    | Prec   | Rec    | F1     | AUC    | Early Stop |
|------|--------|--------|--------|--------|--------|------------|
| 1    | 77.37% | 63.29% | 58.06% | 58.26% | 86.17% | Epoch 10   |
| 2    | 76.20% | 66.37% | 57.89% | 59.45% | 87.03% | Epoch 11   |
| 3    | 76.20% | 56.18% | 56.75% | 55.34% | 86.91% | Epoch 10   |
| 4    | 77.22% | 66.38% | 59.44% | 60.74% | 86.76% | Epoch 10   |
| 5    | 81.27% | 73.58% | 62.46% | 65.07% | 91.26% | Epoch 12   |

## Test Results (Best model from Fold 3)

| Metric    | Baseline | Ours   | Diff    |
|-----------|----------|--------|---------|
| Accuracy  | 74.19%   | 73.38% | -0.81%  |
| Precision | 69.13%   | 64.95% | -4.18%  |
| Recall    | 58.78%   | 60.38% | +1.60%  |
| F1 Score  | 59.95%   | 60.46% | +0.51%  |
| ROC-AUC   | 78.91%   | 92.42% | +13.51% |

## Confusion Matrix

```
[[385  61   2   0   0]
 [  9  70  25   0   1]
 [  5  53 202  11   6]
 [  0   0  26  12   6]
 [  0  17  36   5  56]]
```

## Observations

- Beats baseline on Recall (+1.60%), F1 (+0.51%), and ROC-AUC (+13.51%)
- Accuracy very close to baseline (-0.81%)
- Most stable results across folds (low std)
- Gamma=1.0 provides good balance between minority class focus and overall accuracy
- LR 2e-5 with patience=8 allows sufficient training without overfitting
