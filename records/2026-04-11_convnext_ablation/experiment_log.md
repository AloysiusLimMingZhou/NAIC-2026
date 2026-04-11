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

**Date:** 2026-04-11
**Status:** INVALID — loss function bug meant this actually ran CrossEntropyLoss (same as Exp 1)
**Fix applied:** Added `use_focal_loss` flag to 3-state loss selection. Re-running.

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

## Experiment Order (Remaining)

| Exp | Description | Status |
|-----|-------------|--------|
| 1   | CE baseline | Done |
| 3   | Focal Loss only | Re-running |
| 4   | Focal + Ordinal (0.7/0.3) | Done |
| 5   | + MixUp | Pending |
| 6   | + Three-stage fine-tuning | Pending |
| 7   | + CBAM | Pending |
| 8   | + TTA | Pending |
| 9   | + Snapshot Ensemble | Pending |
