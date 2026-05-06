# Experiment Record: SwinV2-Base Ablation — Ben Graham + TTA

**Date:** 2026-04-27 to 2026-05-03
**Model:** Swin Transformer V2 Base (`microsoft/swinv2-base-patch4-window8-256`)
**Person:** Yi Hui
**Status:** IN PROGRESS (Exp 3 running)

---

## Baseline to Beat

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 74.19% |
| Precision | 69.13% |
| Recall    | 58.78% |
| F1 Score  | 59.95% |
| ROC-AUC   | 78.91% |

**Previous SwinV2 best** (RandomErasing, single fold): 75.61% Acc | **69.80% Recall** | 66.86% F1 | 94.31% ROC-AUC

---

## Ablation Overview

All experiments share these fixed settings:
- SwinV2-Base, **256×256** (native pretrained resolution, up from 224×224)
- **Ben Graham preprocessing** (subtract local Gaussian average — illumination normalisation for multi-camera DR datasets)
- **Snapshot Ensemble + 8-TTA** at inference (all exp results are ensemble, not cherry-picked fold)
- 5-Fold Stratified K-Fold, WeightedRandomSampler
- AdamW, weight_decay=1e-4, CosineAnnealingLR, eta_min=1e-6
- batch_size=16, epochs=20, patience=10, grad_clip=1.0
- Augmentation: Resize(256), HFlip, Rotation(15), ColorJitter, RandomErasing

---

## Experiment 1 — Ben Graham + TTA + Focal(γ=1.0, LS=0.1)

**Hypothesis:** Upgrading to native 256×256 resolution and Ben Graham preprocessing should improve over previous CLAHE+224×224 results.

### Config

| Parameter         | Value                                    |
|-------------------|------------------------------------------|
| Learning rate     | 2e-5                                     |
| Loss              | FocalLoss (gamma=1.0, label_smoothing=0.1) |
| MixUp             | OFF                                      |
| Three-stage       | OFF                                      |
| Warmup            | OFF                                      |
| RandomErasing     | p=0.3                                    |
| TTA               | ON (8 views)                             |

### Test Results (Snapshot Ensemble + 8-TTA)

| Metric    | Baseline | Prev Best | Exp 1      | vs Baseline | vs Prev Best |
|-----------|----------|-----------|------------|-------------|--------------|
| Accuracy  | 74.19%   | 75.61%    | **79.76%** | +5.57% ✅   | +4.15% ✅    |
| Precision | 69.13%   | 67.29%    | **74.87%** | +5.74% ✅   | +7.58% ✅    |
| Recall    | 58.78%   | 69.80%    | 62.50%     | +3.72% ✅   | -7.30% ❌    |
| F1 Score  | 59.95%   | 66.86%    | 62.69%     | +2.74% ✅   | -4.17% ❌    |
| ROC-AUC   | 78.91%   | 94.31%    | 92.79%     | +13.88% ✅  | -1.52%       |

### Confusion Matrix

```
Predicted →    0    1    2    3    4
Actual ↓
Class 0 (448) [419,  27,   2,   0,   0]   Recall: 93.5%
Class 1 (105) [  9,  64,  31,   0,   1]   Recall: 61.0%
Class 2 (277) [  6,  40, 223,   1,   7]   Recall: 80.5%
Class 3  (44) [  0,   0,  29,   4,  11]   Recall:  9.1% ← bottleneck
Class 4 (114) [  2,   7,  27,   0,  78]   Recall: 68.4%
```

### Per-Class Analysis

| Class | Name          | Count | Correct | Recall | Main errors                      |
|-------|---------------|-------|---------|--------|----------------------------------|
| 0     | No DR         | 448   | 419     | 93.5%  | 27 confused with Mild            |
| 1     | Mild          | 105   | 64      | 61.0%  | 31 confused with Moderate        |
| 2     | Moderate      | 277   | 223     | 80.5%  | 40 confused with Mild            |
| 3     | Severe        | 44    | 4       | 9.1%   | 29 confused with Moderate, 11 with Prolif |
| 4     | Proliferative | 114   | 78      | 68.4%  | 27 confused with Moderate        |

### Observations
- Accuracy significantly improved (+5.57% vs baseline, +4.15% vs prev best) — 256×256 + Ben Graham helps overall discrimination
- Recall lower than previous best but **that was a cherry-picked Fold 5 result** — ensemble is a fairer comparison
- Class 3 (Severe) catastrophically bad at 9.1% — model almost never predicts Severe correctly
- Class 0 (No DR) excellent at 93.5% — easy majority class
- Focal Loss without MixUp: good accuracy but not enough minority class boost

---

## Experiment 2 — CE+LS(0.03) + Soft-label MixUp(α=0.3)

**Hypothesis:** MixUp added +1.24% Recall on ConvNeXt. Switching from Focal to CE+LS(0.03) since MixUp + Focal Loss conflicts (soft targets break the focal modulation factor).

### Config

| Parameter         | Value                                    |
|-------------------|------------------------------------------|
| Learning rate     | 1e-4                                     |
| Loss              | CrossEntropyLoss + LabelSmoothing(ε=0.03) |
| MixUp             | ON — soft-label, α=0.3, 50% of batches  |
| Three-stage       | OFF                                      |
| Warmup            | OFF                                      |
| RandomErasing     | p=0.5 (bumped from 0.3 to match prev best) |
| TTA               | ON (8 views)                             |

### Test Results (Snapshot Ensemble + 8-TTA)

| Metric    | Baseline | Exp 1  | Exp 2      | vs Exp 1 |
|-----------|----------|--------|------------|----------|
| Accuracy  | 74.19%   | 79.76% | 79.05%     | -0.71%   |
| Precision | 69.13%   | 74.87% | 72.67%     | -2.20%   |
| Recall    | 58.78%   | 62.50% | 60.01%     | -2.49% ❌ |
| F1 Score  | 59.95%   | 62.69% | 60.34%     | -2.35% ❌ |
| ROC-AUC   | 78.91%   | 92.79% | **92.94%** | +0.15%   |

### Confusion Matrix

```
Predicted →    0    1    2    3    4
Actual ↓
Class 0 (448) [401,  30,  16,   0,   1]   Recall: 89.5%
Class 1 (105) [  5,  46,  53,   0,   1]   Recall: 43.8% ← collapsed
Class 2 (277) [  5,  16, 250,   0,   6]   Recall: 90.3% ← improved
Class 3  (44) [  0,   0,  35,   2,   7]   Recall:  4.5% ← worse
Class 4 (114) [  0,   1,  30,   1,  82]   Recall: 71.9% ← slightly better
```

### Per-Class vs Exp 1

| Class | Exp 1 Recall | Exp 2 Recall | Change |
|-------|-------------|-------------|--------|
| 0     | 93.5%       | 89.5%       | -4.0% ❌ |
| 1     | 61.0%       | 43.8%       | **-17.2% ❌** |
| 2     | 80.5%       | 90.3%       | +9.8% ✅ |
| 3     | 9.1%        | 4.5%        | -4.6% ❌ |
| 4     | 68.4%       | 71.9%       | +3.5% ✅ |

### Observations
- **MixUp does NOT transfer from ConvNeXt to SwinV2** — hurts net Recall despite helping Class 2
- Root cause: severity-aware MixUp blends 0↔1 and 1↔2. Makes model better at Moderate (Class 2) but collapses Mild (Class 1) recall from 61% → 44%
- Class 1 (Mild) is the biggest casualty — blended so heavily with Moderate that model can no longer distinguish them
- Class 3 (Severe) gets worse still — MixUp offers no benefit here; it's a volume problem (only 44 samples)
- **Conclusion: Drop MixUp for SwinV2. Return to Focal+LS baseline.**

---

## Experiment 3 — ECA + EMA (on Exp 1 base)

**Status:** Running

**Config:** Exp 1 base (Focal γ=1.0 + LS=0.1, lr=2e-5, no MixUp) + ECA channel attention (1024-dim) + EMA inference (decay=0.9998)

**Hypothesis:**
- ECA was +5% F1 / +3.5% Recall on ConvNeXt — testing transfer to SwinV2's 1024-dim pooled output
- EMA (shadow weight averaging) is particularly effective on small datasets (<5k images) — used in DeiT/BEiT
- Key goal: improve Class 3 (Severe) recall from 9.1% — ECA's channel attention may help distinguish lesion severity features

---

## Summary So Far

| Experiment | Key Change | Recall | F1     | Winner? |
|------------|-----------|--------|--------|---------|
| Baseline   | —         | 58.78% | 59.95% | —       |
| Exp 1      | Ben Graham + 256×256 + TTA | **62.50%** | **62.69%** | ✅ Best |
| Exp 2      | + CE+LS + MixUp | 60.01% | 60.34% | ❌ Worse |
| Exp 3      | Exp 1 + ECA + EMA | TBD | TBD | Running |

**Key findings:**
1. **Ben Graham + 256×256 + TTA** — solid baseline improvement (+5.57% Acc, +3.72% Recall vs original baseline)
2. **MixUp hurts SwinV2** — collapses Class 1 recall (-17.2%) while only helping Class 2
3. **Class 3 (Severe)** is the persistent bottleneck — 44 samples, never predicted reliably
4. Ensemble scoring is more honest than cherry-picked single fold — previous "69.80% Recall" was Fold 5 only
