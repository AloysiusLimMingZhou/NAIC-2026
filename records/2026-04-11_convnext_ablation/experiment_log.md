# ConvNeXt-Small Ablation Experiment Log

**Model:** ConvNeXt-Small (`facebook/convnext-small-224`)
**Dataset:** ~4,939 DR images, 5-class severity classification
**Training:** Stratified 5-Fold, WeightedRandomSampler, CLAHE + Retina Cropping, Early Stopping (patience=10)
**Base Config:** lr=1e-4, AdamW, CosineAnnealing, batch_size=32, epochs=30

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

## Experiment 1 — CrossEntropyLoss Baseline (ConvNeXt)

**Date:** 2026-04-11
**Loss:** CrossEntropyLoss
**Feature Flags:** All False

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 77.33% |
| Precision | 77.36% |
| Recall    | 59.39% |
| F1 Score  | 59.66% |
| ROC-AUC   | N/A (bug) |

**Confusion Matrix:**
```
[[403  44   1   0   0]
 [ 14  58  32   0   1]
 [  8  30 229   0  10]
 [  0   0  36   3   5]
 [  0   6  37   0  71]]
```

**Notes:**
- Beats baseline on Accuracy (+3.14%) and Precision (+8.23%)
- Recall slightly up (+0.61%), F1 essentially flat (-0.29%)
- Class 3 (Severe) only 3/44 correct (6.8%) — major bottleneck
- Heavy overfitting: train acc ~95% vs val acc ~60-74%
- Early stopping at epoch 10-13 across folds
- ROC-AUC bug: softmax on float16 outputs, fixed after this run

---

## Experiment 3 — Focal Loss Only

**Date:** 2026-04-12
**Loss:** FocalLoss(gamma=2.0)
**Feature Flags:** use_focal_loss=True, all others False
**Note:** First attempt (2026-04-11) was invalid due to missing `use_focal_loss` branch — actually ran CE. Re-ran after fix.

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 70.45% |
| Precision | 70.35% |
| Recall    | 57.26% |
| F1 Score  | 55.54% |
| ROC-AUC   | N/A (bug — ensemble prob normalization, fixed after this run) |

**Confusion Matrix:**
```
[[327 119   2   0   0]
 [  4  74  26   0   1]
 [  4  42 230   0   1]
 [  0   0  36   2   6]
 [  0   6  44   1  63]]
```

**Notes:**
- Focal Loss alone **hurts** — worse than both CE and Focal+Ordinal across all metrics
- Class 0 recall dropped heavily: 90% (CE) -> 73% (Focal) — over-reweighting toward hard examples
- Class 1 best so far at 70% (up from 55% CE), but net effect is negative
- Class 3 still dismal: 2/44 (5%)
- Severe overfitting persists: train acc ~96% vs val acc ~58-71%
- Conclusion: Focal Loss alone is too aggressive without ordinal penalty to guide it

---

## Experiment 4 — Focal Loss + Ordinal Penalty (0.7/0.3)

**Date:** 2026-04-11
**Loss:** CombinedLoss(FocalLoss(gamma=2.0) * 0.7 + OrdinalPenaltyLoss * 0.3)
**Feature Flags:** use_ordinal_loss=True, all others False

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 74.60% |
| Precision | 73.32% |
| Recall    | 59.92% |
| F1 Score  | 59.32% |
| ROC-AUC   | 0.00% (bug — float16 softmax, fixed after this run) |

**Confusion Matrix:**
```
[[364  83   1   0   0]
 [  3  71  30   0   1]
 [  5  33 232   1   6]
 [  0   0  34   4   6]
 [  0   3  45   0  66]]
```

**Notes:**
- Recall best so far at 59.92% (+1.14% vs baseline)
- Class 1 improved significantly: 55% -> 68% recall
- Class 3 still poor: 4/44 (9%), up from 3/44
- Accuracy dropped vs Exp 1 (74.60% vs 77.33%) — ordinal penalty trades Class 0 accuracy for minority class recall
- Class 0 recall dropped: 90% -> 81% (more Class 0 misclassified as Class 1)
- More severe overfitting than Exp 1

---

## Experiment 5 — CE + Severity-aware MixUp

**Date:** 2026-04-12
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, all others False

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 77.23% |
| Precision | 77.49% |
| Recall    | **60.02%** |
| F1 Score  | 59.41% |
| ROC-AUC   | 93.58% |

**Confusion Matrix:**
```
[[405  42   1   0   0]
 [ 13  65  25   0   2]
 [ 10  41 217   0   9]
 [  0   0  36   2   6]
 [  0   7  33   0  74]]
```

**Notes:**
- New best Recall: 60.02% (+1.24% vs baseline)
- Overfitting dramatically reduced: train acc ~80% vs val ~70% (previously 95% vs 60-74%)
- Early stopping at epoch 15-16 (vs 10-13 before) — MixUp stabilizes training
- ROC-AUC finally working: 93.58% — excellent discrimination ability
- Class 3 (Severe) still terrible: 2/44 (4.5%)
- Class 4 best so far: 74/114 (65%)
- Conclusion: MixUp is the best regularizer found so far — keep it for remaining experiments

---

## Experiment Order (Completed)

| Exp | Description | Status |
|-----|-------------|--------|
| 1   | CE baseline | Done |
| 3   | Focal Loss only | Done |
| 4   | Focal + Ordinal (0.7/0.3) | Done |
| 5   | CE + MixUp | Done ← Best so far (Recall 60.02%, ROC-AUC 93.58%) |

---

## Experiment 6 — CE + MixUp + Three-Stage Fine-Tuning

**Date:** 2026-04-12
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, use_three_stage=True, all others False

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 73.68% |
| Precision | 69.98% |
| Recall    | 57.77% |
| F1 Score  | 58.46% |
| ROC-AUC   | 92.13% |

**Confusion Matrix:**
```
[[418  30   0   0   0]
 [ 21  69  12   0   3]
 [ 13  74 177   1  12]
 [  0   1  33   7   3]
 [  1  11  44   1  57]]
```

**Notes:**
- Three-stage fine-tuning **HURTS** — regresses across all metrics vs Exp 5
  - Accuracy: -3.55% (77.23% → 73.68%)
  - Recall: -2.25% (60.02% → 57.77%)
  - F1: -0.95% (59.41% → 58.46%)
  - ROC-AUC: -1.45% (93.58% → 92.13%)
- Class 0 recall improved slightly (91.46%), but Class 1-3 all dropped
- Stage 1 (head-only, 5 epochs) may have been too aggressive; unfreezing at epoch 5 introduced instability
- Discriminative LR schedule (1e-6, 1e-5, 5e-6) didn't help — may need tuning or removal
- **Conclusion:** Skip three-stage fine-tuning. Exp 5 (CE+MixUp baseline) is superior.

---

## Experiment 6b — CE + MixUp + Class-Weighted CE

**Date:** 2026-04-12
**Loss:** WeightedCrossEntropyLoss (inverse class frequency weights)
**Feature Flags:** use_mixup=True, all others False

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 66.50% |
| Precision | 62.57% |
| Recall    | 57.42% |
| F1 Score  | 54.52% |
| ROC-AUC   | 93.21% |

**Confusion Matrix:**
```
[[285 162   1   0   0]
 [  3  78  22   0   2]
 [  1  52 216   1   7]
 [  0   0  36   2   6]
 [  0   7  28   3  76]]
```

**Notes:**
- Weighted CE **severely HURTS** — catastrophic regression
  - Accuracy: -10.73% (77.23% → 66.50%) 
  - Precision: -14.92% (77.49% → 62.57%) 
  - F1: -4.89% (59.41% → 54.52%) 
  - Recall: -2.60% (60.02% → 57.42%)
  - ROC-AUC: -0.37% (93.58% → 93.21%)
- Class 0 recall **collapsed**: 91.46% (Exp 5) → 63.6% (Exp 6b) — 162 false positives
- Model learned to over-predict minority classes at the expense of majority class
- Inverse frequency weighting too aggressive; penalty dominated loss function
- **Conclusion:** Skip class-weighted CE. Exp 5 (CE+MixUp baseline) is superior.

---

## Experiment 7 — CE + MixUp + ECA Attention

**Date:** 2026-04-12
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, use_eca=True, all others False

| Metric    | Score  |
|-----------|--------|
| Accuracy  | **78.85%** |
| Precision | **79.29%** |
| Recall    | **63.58%** |
| F1 Score  | **64.46%** |
| ROC-AUC   | **93.78%** |

**Confusion Matrix:**
```
[[401  45   2   0   0]
 [  9  70  24   0   2]
 [  8  36 229   0   4]
 [  0   0  30   7   7]
 [  0   8  34   0  72]]
```

**Notes:**
- NEW BEST across ALL metrics — biggest jump yet
  - Accuracy: +1.62% (77.23% → 78.85%)
  - Precision: +1.80% (77.49% → 79.29%)
  - Recall: +3.56% (60.02% → 63.58%)
  - F1: +5.05% (59.41% → 64.46%)
  - ROC-AUC: +0.20% (93.58% → 93.78%)
- Class 3 (Severe) dramatically improved: 2/44 (4.5%) → 7/44 (15.9%) — 3.5x improvement
- Class 4 (Proliferative): 72/114 (63.2%), consistent with Exp 5
- Class 1 (Mild): 70/105 (66.7%), up from 62% in Exp 5
- Class 2 (Moderate): 229/277 (82.7%), up from 78.3% in Exp 5
- ECA adds ~5 parameters — effectively free regularization with attention guidance
- **Conclusion:** ECA is a clear winner. Keep for all future experiments. Proceed to test CBAM (Exp 7b).

---

## Experiment 7b — CE + MixUp + CBAM (replace ECA)

**Date:** 2026-04-12
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, use_cbam=True, use_eca=False, all others False

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 78.04% |
| Precision | 74.14% |
| Recall    | 62.94% |
| F1 Score  | 61.80% |
| ROC-AUC   | 94.28% |

**Confusion Matrix:**
```
[[389  56   3   0   0]
 [  6  77  20   0   2]
 [  7  36 227   0   7]
 [  0   0  35   3   6]
 [  0   6  32   1  75]]
```

**Notes:**
- CBAM **loses to ECA** across almost all metrics
  - Accuracy: -0.81% (78.85% → 78.04%)
  - Precision: -5.15% (79.29% → 74.14%)
  - F1: -2.66% (64.46% → 61.80%)
  - Recall: -0.64% (63.58% → 62.94%)
  - ROC-AUC: +0.50% (93.78% → 94.28%) — only improvement
- Class 3 (Severe) regressed: 7/44 (15.9%) → 3/44 (6.8%) — back to Exp 1 levels
- Class 1 (Mild): 77/105 (73.3%) — best so far, up from 66.7% (ECA)
- Training unstable: val loss spiking to 1.5–1.8 across folds
- CBAM's heavier parameterization (~thousands of params vs ~5) causes more overfitting on small dataset
- Early stopping triggered earlier on average (epoch 11–16)
- **Conclusion:** CBAM is worse than ECA. Stick with ECA for all future experiments.

---

## Experiment 8 — CE + MixUp + ECA + Warmup

**Date:** 2026-04-12
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, use_eca=True, use_warmup=True (3 epochs, LinearLR start_factor=0.01), all others False

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 77.13% |
| Precision | 77.75% |
| Recall    | 62.19% |
| F1 Score  | 61.02% |
| ROC-AUC   | 93.72% |

**Confusion Matrix:**
```
[[381  66   1   0   0]
 [  7  69  27   0   2]
 [  4  34 228   0  11]
 [  0   0  34   3   7]
 [  0   2  31   0  81]]
```

**Notes:**
- Warmup **HURTS** — regresses across all metrics vs Exp 7
  - Accuracy: -1.72% (78.85% → 77.13%)
  - Precision: -1.54% (79.29% → 77.75%)
  - F1: -3.44% (64.46% → 61.02%)
  - Recall: -1.39% (63.58% → 62.19%)
  - ROC-AUC: -0.06% (93.78% → 93.72%)
- Class 3 (Severe) regressed: 7/44 (15.9%) → 3/44 (6.8%)
- Class 4 (Proliferative) improved: 72/114 → 81/114 (71.1%) — best so far
- First 3 epochs wasted at very low LR (1e-6 → 6.7e-5), delaying convergence
- With early stopping patience=10, model runs out of runway — peaks lower, overfits before recovering
- Val loss unstable: spiking to 1.5–1.9 in later epochs across folds
- **Conclusion:** Skip warmup. Exp 7 (CE+MixUp+ECA) remains best.

---

## Experiment 9 — CE + MixUp + ECA + Discriminative LR

**Date:** 2026-04-13
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, use_eca=True, use_discriminative_lr=True, all others False
**LR Schedule:** Early layers 1e-6, Late backbone 1e-5, Head+ECA 1e-4

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 73.18% |
| Precision | 70.20% |
| Recall    | 57.86% |
| F1 Score  | 57.96% |
| ROC-AUC   | 92.43% |

**Confusion Matrix:**
```
[[383  65   0   0   0]
 [ 12  74  18   0   1]
 [ 12  51 210   2   2]
 [  0   1  32   6   5]
 [  0  13  50   1  50]]
```

**Notes:**
- Discriminative LR **CATASTROPHIC** — worst result since Exp 6b, below baseline on Acc and Recall
  - Accuracy: -5.67% (78.85% → 73.18%)
  - Precision: -9.09% (79.29% → 70.20%)
  - F1: -6.50% (64.46% → 57.96%)
  - Recall: -5.72% (63.58% → 57.86%)
  - ROC-AUC: -1.35% (93.78% → 92.43%)
- Backbone starved: early layers at 1e-6 barely learn, cosine decay from 1e-6 stays flat
- Printed LR stuck at 1e-6 (param_groups[0]) — misleading but confirms backbone undertrained
- Class 4 collapsed: 72/114 (63.2%) → 50/114 (43.9%)
- Class 3 slightly up: 7/44 → 6/44, but meaningless given overall regression
- Folds 3-5 ran to epoch 20-26 without finding good F1 — model never converges properly
- **Conclusion:** Discriminative LR at 100:10:1 ratio starves backbone. Skip entirely.

---

## Experiment 10a — CE + MixUp + ECA + Resolution 384×384

**Date:** 2026-04-13
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, use_eca=True, all others False
**Resolution:** 384×384 (was 224×224), batch_size=16 (was 32)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 71.96% |
| Precision | 70.28% |
| Recall    | 61.83% |
| F1 Score  | 59.56% |
| ROC-AUC   | 93.73% |

**Confusion Matrix:**
```
[[310 133   5   0   0]
 [  3  73  27   0   2]
 [  3  26 238   1   9]
 [  0   0  33   4   7]
 [  0   4  23   1  86]]
```

**Notes:**
- Resolution increase **HURTS** — major accuracy regression
  - Accuracy: -6.89% (78.85% → 71.96%)
  - Precision: -9.01% (79.29% → 70.28%)
  - F1: -4.90% (64.46% → 59.56%)
  - Recall: -1.75% (63.58% → 61.83%)
  - ROC-AUC: -0.05% (93.78% → 93.73%)
- Class 0 collapsed: 401/448 (89.5%) → 310/448 (69.2%) — 133 misclassified as Class 1
- Class 4 (Proliferative) best ever: 86/114 (75.4%), up from 72/114 (63.2%)
- Class 2 (Moderate) improved: 229/277 → 238/277 (85.9%)
- Class 3 (Severe) regressed: 7/44 → 4/44 (9.1%)
- Severe overfitting: train acc ~85-89% vs val ~62-67%
- Pretrained weights optimized for 224 — position embeddings misaligned at 384
- Half batch size (16 vs 32) may also destabilize BatchNorm/training dynamics
- **Conclusion:** Resolution 384 hurts. Stick with 224×224.

---

## Experiment 10b — CE(label_smoothing=0.1) + MixUp + ECA

**Date:** 2026-04-13
**Loss:** CrossEntropyLoss(label_smoothing=0.1)
**Feature Flags:** use_mixup=True, use_eca=True, all others False
**Resolution:** 224×224, batch_size=32

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 77.43% |
| Precision | 79.45% |
| Recall    | 62.55% |
| F1 Score  | 61.97% |
| ROC-AUC   | 92.80% |

**Confusion Matrix:**
```
[[382  65   1   0   0]
 [  7  76  20   0   2]
 [  6  37 232   0   2]
 [  0   0  34   4   6]
 [  0   7  36   0  71]]
```

**Notes:**
- Label smoothing **HURTS** — mild regression across most metrics
  - Accuracy: -1.42% (78.85% → 77.43%)
  - F1: -2.49% (64.46% → 61.97%)
  - Recall: -1.03% (63.58% → 62.55%)
  - ROC-AUC: -0.98% (93.78% → 92.80%)
  - Precision: +0.16% (79.29% → 79.45%) — only improvement
- Class 3 (Severe) regressed: 7/44 → 4/44 (9.1%)
- ROC-AUC degraded significantly during training (0.90 → 0.77-0.82 by later epochs)
- Label smoothing flattens probability distribution, hurting confident discrimination
- Combined with MixUp (which already softens targets), double-smoothing is too much
- **Conclusion:** Label smoothing hurts when MixUp is already active. Skip.

---

## Experiment 11 — CE(0.65) + BoundaryOrdinal(0.35) + MixUp + ECA

**Date:** 2026-04-14
**Loss:** 0.65 * CrossEntropyLoss + 0.35 * BoundaryAwareOrdinalLoss (3× penalty on 3→2, 2× on 2→3/3→4/4→3)
**Feature Flags:** use_mixup=True, use_eca=True, all others False

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 78.24% |
| Precision | 72.82% |
| Recall    | 61.04% |
| F1 Score  | 61.42% |
| ROC-AUC   | 93.57% |

**Confusion Matrix:**
```
[[393  51   4   0   0]
 [  8  67  28   0   2]
 [  8  19 245   1   4]
 [  0   0  35   4   5]
 [  0   5  44   1  64]]
```

**Notes:**
- Boundary Ordinal **does NOT beat Exp 7** — regresses across all metrics
  - Accuracy: -0.61% (78.85% → 78.24%)
  - Precision: -6.47% (79.29% → 72.82%)
  - F1: -3.04% (64.46% → 61.42%)
  - Recall: -2.54% (63.58% → 61.04%)
  - ROC-AUC: -0.21% (93.78% → 93.57%)
- Class 2 best ever: 245/277 (88.4%) — boundary penalty improved moderate classification
- Class 3 (Severe) regressed: 7/44 → 4/44 (9.1%)
- Class 4 collapsed: 72/114 (63.2%) → 64/114 (56.1%) — boundary penalty overcorrected
- Early stopping triggered very early (fold 2: ep 10, fold 3: ep 11, fold 5: ep 10)
- Ordinal loss component causes faster convergence but lower peak performance
- **Conclusion:** Boundary ordinal loss overcorrects. Exp 7 is the definitive solo ceiling.

---

## Final Summary — Ablation Complete

**BEST SOLO MODEL: Experiment 7 — CE + MixUp + ECA**

| Metric    | Baseline | Exp 7 (Best) | Improvement |
|-----------|----------|--------------|-------------|
| Accuracy  | 74.19%   | **78.85%**   | +4.66%      |
| Precision | 69.13%   | **79.29%**   | +10.16%     |
| Recall    | 58.78%   | **63.58%**   | +4.80%      |
| F1 Score  | 59.95%   | **64.46%**   | +4.51%      |
| ROC-AUC   | 78.91%   | **93.78%**   | +14.87%     |

**11 experiments tested, 2 techniques proven:**
1. Severity-aware MixUp — reduces overfitting, stabilizes training
2. ECA (Efficient Channel Attention) — ~5 params, guides feature selection

**9 techniques that HURT:** Focal Loss, Weighted CE, CBAM, Three-stage fine-tuning, Warmup, Discriminative LR, Resolution 384, Label Smoothing, Boundary Ordinal Loss

**Next step:** Ensemble Exp 7 ConvNeXt (0.4) + Aloysius's EfficientNet-B3 (0.6)

---

## Experiment 12 — CE + MixUp + ECA + Dropout(0.3)

**Date:** 2026-05-08
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, use_eca=True, all others False
**Change vs Exp 7:** Added `nn.Dropout(p=0.3)` after ECA global avg pool, before classifier head

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 77.53% |
| Precision | 78.79% |
| Recall    | 61.11% |
| F1 Score  | 60.84% |
| ROC-AUC   | 93.68% |

**Confusion Matrix:**
```
[[384  60   4   0   0]
 [ 10  66  28   0   1]
 [  6  25 241   0   5]
 [  0   0  35   3   6]
 [  0   4  38   0  72]]
```

**Notes:**
- Dropout 0.3 **HURTS** — regresses across all metrics vs Exp 7 (verified on Kaggle: notebookb64761cbf4.ipynb)
  - Accuracy: -1.32% (78.85% → 77.53%)
  - Recall: -2.47% (63.58% → 61.11%)
  - F1: -3.62% (64.46% → 60.84%)
  - ROC-AUC: -0.10% (93.78% → 93.68%)
  - Precision: -0.50% (79.29% → 78.79%)
- Class 2 (Moderate) improved: 229/277 → 241/277 (87.0%) — dropout helped boundary generalisation here
- Class 3 (Severe): 3/44 (6.8%) — regressed from 7/44 (15.9%) in Exp 7
- Class 0 improved: 364→384 (85.7% vs 89.5% Exp 7 — still worse)
- Val loss spiked during training (1.5–2.5) indicating miscalibration, hurting softmax ensemble averaging
- Root cause: dropout=0.3 with MixUp+weight_decay=0.01 already regularizing → triple regularization is too much

---

## Experiment 13 — CE + MixUp + ECA + Dropout(0.1)

**Date:** 2026-05-08
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, use_eca=True, all others False
**Change vs Exp 12:** Reduced dropout from 0.3 → 0.1 (lighter regularization)

---

## Experiment 14 — CE + MixUp + ECA + Gradient Accumulation (effective batch 64)

**Date:** 2026-05-09
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, use_eca=True, all others False
**Change vs Exp 7:** Added gradient accumulation (ACCUMULATION_STEPS=2), effective batch size 32×2=64. dropout_rate=0.0

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 77.02% |
| Precision | 73.49% |
| Recall    | 60.08% |
| F1 Score  | 61.02% |
| ROC-AUC   | 93.32% |

**Per-Fold Validation F1:**
| Fold | Acc    | Recall | F1     | ROC-AUC |
|------|--------|--------|--------|---------|
| 1    | 72.82% | 52.39% | 52.05% | 86.93%  |
| 2    | 69.11% | 52.74% | 51.99% | 88.61%  |
| 3    | 63.67% | 49.04% | 47.15% | 86.58%  |
| 4    | 64.43% | 48.16% | 48.27% | 82.78%  |
| 5    | 63.29% | 53.12% | 51.64% | 87.14%  |

**Confusion Matrix (Test Ensemble + TTA):**
```
[[398  50   0   0   0]
 [  5  67  31   0   2]
 [  8  30 233   0   6]
 [  0   0  36   6   2]
 [  0   6  49   2  57]]
```

**Notes:**
- Gradient accumulation **HURTS** — regresses across all metrics vs Exp 7
  - Accuracy: -1.83% (78.85% → 77.02%)
  - Precision: -5.80% (79.29% → 73.49%)
  - Recall: -3.50% (63.58% → 60.08%)
  - F1: -3.44% (64.46% → 61.02%)
  - ROC-AUC: -0.46% (93.78% → 93.32%)
- Early stopping hit early across all folds (ep 10–16), consistent with underfitting
- Validation F1 per fold very low (47–52%) vs training suggesting high variance
- Larger effective batch (64) flattens gradients — loss landscape too smooth, optimizer overshoots
- Accumulation doubles steps-per-update but halves update frequency; with LR=1e-4 tuned for batch=32, this is effectively too high per effective batch
- Class 3 (Severe): 6/44 (13.6%) — slightly lower than Exp 7 (7/44)
- Class 4 (Proliferative): 57/114 (50.0%) — significant regression from Exp 7 (72/114, 63.2%)
- **Conclusion:** Gradient accumulation with default LR hurts. Would need LR scaling (×2) to compensate. Skip for now.

---

## Experiment 15 — CE + MixUp + ECA + SGD+Nesterov (vs AdamW)

**Date:** 2026-05-09
**Loss:** CrossEntropyLoss
**Feature Flags:** use_mixup=True, use_eca=True, all others False
**Change vs Exp 7:** Replaced AdamW with SGD(lr=1e-3, momentum=0.9, nesterov=True). Local macOS run (num_workers=0, pin_memory=False)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 62.85% |
| Precision | 69.18% |
| Recall    | 48.59% |
| F1 Score  | 46.37% |
| ROC-AUC   | 87.67% |

**Per-Fold Training Behavior:**
| Fold | Epochs | Train Acc (final) | Val Acc (best) |
|------|--------|------------------|----------------|
| 1    | 16     | ~57%             | ~67%           |
| 2    | 16     | ~57%             | ~56%           |
| 3    | 18     | ~57%             | ~58%           |
| 4    | 23     | ~59%             | ~61%           |
| 5    | 19     | ~58%             | ~57%           |

**Confusion Matrix (Test Ensemble + TTA):**
```
[[380  62   6   0   0]
 [ 18  78   9   0   0]
 [ 19 124 120   0  14]
 [  0  10  28   2   4]
 [  4  35  34   0  41]]
```

**Notes:**
- SGD **CATASTROPHIC** — worst result in entire ablation study
  - Accuracy: -16.00% (78.85% → 62.85%)
  - Precision: -10.11% (79.29% → 69.18%)
  - Recall: -14.99% (63.58% → 48.59%)
  - F1: -18.09% (64.46% → 46.37%)
  - ROC-AUC: -6.11% (93.78% → 87.67%)
- Class 0 (Healthy) badly overfitted: 380/448 (84.8%) correct but precision only 69% (many FP from other classes)
- Class 2 (Moderate) collapsed: 120/277 (43.3%) recall — 124 misclassified as Class 1
- Class 3 (Severe) essentially failed: 2/44 (4.5%) — worse than baseline
- Class 4 (Proliferative) severely regressed: 41/114 (36%) vs 72/114 (63.2%) in Exp 7
- Training loss plateaued early (epoch 6–10), val loss highly unstable across folds (spiking to 1.4–1.5)
- SGD learning rate 1e-3 too aggressive for this model/task; momentum+nesterov couldn't stabilize convergence
- Early stopping triggered early (ep 16–23) across all folds, indicating poor learning dynamics
- **Root cause:** SGD with lr=1e-3 (tuned for batch=32) caused oscillations; no adaptive scaling like AdamW. Would need extensive LR tuning per fold.
- **Conclusion:** SGD is fundamentally worse than AdamW for this task. Skip entirely. Exp 7 (AdamW+MixUp+ECA) remains definitive solo best.
