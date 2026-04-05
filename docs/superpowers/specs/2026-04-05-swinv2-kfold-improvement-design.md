# SwinV2 Improvement Design — Stratified K-Fold + Augmentation + Focal Loss

**Date:** 2026-04-05
**Author:** Yi Hui
**Model:** Swin Transformer V2 (`microsoft/swinv2-base-patch4-window8-256`)
**Goal:** Beat baseline recall (58.78%) as primary metric, then accuracy and F1

---

## Baseline to Beat

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 74.19% |
| Precision | 69.13% |
| Recall    | 58.78% |
| F1 Score  | 59.95% |
| ROC-AUC   | 78.91% |

---

## 1. Model Swap

- Replace ViT with `Swinv2ForImageClassification`
- Pretrained: `microsoft/swinv2-base-patch4-window8-256`
- Input resolution: **224x224** (for cross-model consistency; dataset images are already 224x224)
- `AutoImageProcessor.from_pretrained` loads SwinV2-specific mean/std for normalization

---

## 2. Data Splitting — Stratified K-Fold

**Rationale:** Judge explicitly stressed Stratified K-Fold using scikit-learn.

**Strategy:** Hold-out test set + 5-Fold CV on the rest.

1. Split off **20% as a fixed test set** using `train_test_split(stratify=df['Label'], random_state=42)`
2. On the remaining **80%**, apply `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
3. Each fold: 4/5 train (64% of total), 1/5 validation (16% of total)
4. Train 5 independent models (fresh weights each fold)
5. Report **average metrics +/- std** across all 5 folds
6. Best fold model (lowest val loss) gets evaluated on the held-out test set for final metrics

---

## 3. Data Augmentation

**Training transforms** (applied after CLAHE):

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])
```

**Validation/Test transforms** (no augmentation, no CLAHE):

```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])
```

Conservative augmentation for medical images. Each parameter is tunable individually later.

---

## 4. Focal Loss

Replace `CrossEntropyLoss` with a custom `FocalLoss` module.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, label_smoothing=0.1):
        # alpha: per-class weights from compute_class_weight('balanced')
        # gamma: focusing parameter (higher = more focus on hard samples)
        # label_smoothing: same as baseline (0.1)
        # Implementation: use F.cross_entropy with label_smoothing to get
        # the base CE loss and log-probabilities, then apply focal modulation
        # factor (1 - p_t)^gamma on top.
```

- `alpha` = class weights from `compute_class_weight('balanced')` — recomputed per fold
- `gamma` = 2.0 (default, tunable: try 1.0 or 3.0 later)
- `label_smoothing` = 0.1 (same as baseline)

**Why Focal Loss:** Standard CE treats all samples equally. Focal Loss down-weights easy/well-classified samples and focuses on hard ones — directly targets improving recall on minority classes.

---

## 5. WeightedRandomSampler

Add `WeightedRandomSampler` to the training DataLoader per fold.

- Compute sample weights: `weight[i] = 1.0 / class_count[label[i]]`
- Replaces `shuffle=True` in `DataLoader` (sampler handles randomization)
- Recomputed per fold since train split changes

**Why:** Focal Loss adjusts the loss gradient, but the model still sees the original class distribution per epoch. The sampler ensures minority classes appear more frequently during training. They complement each other.

---

## 6. Hyperparameters (Unchanged from Baseline)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Batch size | 16 |
| Epochs per fold | 20 |
| Early stopping patience | 5 |
| Gradient clipping | max_norm=1.0 |

These are the same as baseline. Tuning comes in later iterations.

---

## 7. Experiment Tracking

**ExperimentTracker** modifications:

- Log config once, including `"k_folds": 5`
- Log per-fold validation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Log aggregated metrics: mean +/- std across 5 folds
- Log final test metrics from the best fold's model
- Save only the best model (lowest val loss across all folds)

---

## 8. Notebook Updates

Both notebooks updated with identical training logic:

| Notebook | Device | AMP |
|----------|--------|-----|
| `baseline_training_macos.ipynb` | MPS | No autocast/GradScaler |
| `baseline_training.ipynb` | CUDA | autocast + GradScaler |

**Primary development** on macOS notebook (Yi Hui's machine). CUDA notebook synced for teammates.

---

## 9. Iterative Tuning Plan (Post-Implementation)

After the base improvements are validated, tune one variable at a time:

1. Augmentation strength (rotation degrees, jitter values)
2. Learning rate (3e-5, 1e-4)
3. Focal Loss gamma (1.0, 3.0)
4. Input resolution (256x256 — SwinV2's native)
5. Batch size (8, 32)

Each change = one experiment run, compare against the K-Fold baseline.

---

## 10. Success Criteria

- All 5 metrics beat the baseline table
- Recall (primary) exceeds 58.78%
- K-Fold average metrics reported with standard deviation
- Clean experiment records saved to `records/`
