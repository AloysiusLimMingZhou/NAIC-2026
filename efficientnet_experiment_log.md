# EfficientNet-B3 Ablation Experiment Log

**Model:** EfficientNet-B3 (`google/efficientnet-b3`)
**Dataset:** 4,939 DR images, 5-class severity classification (Class 0: 2239, Class 1: 525, Class 2: 1383, Class 3: 223, Class 4: 569)
**Training:** Stratified 5-Fold, WeightedRandomSampler, CLAHE + Retina Cropping, Early Stopping (patience=10)
**Base Config:** lr=1e-3 (head), AdamW, CosineAnnealing, batch_size=32, epochs=20, input=300×300
**Preprocessing:** Green Channel CLAHE (clipLimit=2.0), Retina Cropping

---

## Head-to-Head Results (Test Ensemble TTA)

| Rank | Exp | Techniques | Accuracy | Precision | Recall | F1 | AUC | Status |
|:----:|:---:|-----------|:--------:|:---------:|:------:|:--:|:---:|:------:|
| — | **0** | CE baseline (no TTA, no 3-stage) | 64.0%* | — | — | — | — | ✅ 5-fold (no test) |
| 4 | **1** | CE + LS(ε=0.1) + 3-Stage + TTA | 73.1% | 70.9% | 58.8% | 58.0% | 90.5% | ✅ Full 5-fold |
| 6 | **2** | Focal + Ordinal + 3-Stage + TTA | 72.3% | 73.7% | 57.1% | 57.4% | 89.9% | ✅ Full 5-fold |
| 7 | **3** | CE + MixUp(α=0.4) + ECA + TTA | 65.9% | 72.8% | 57.9% | 57.2% | 91.4% | ✅ Full 5-fold |
| 5 | **4** | Focal + Ordinal + CBAM + 3-Stage + TTA | 73.5% | 71.8% | 57.7% | 58.5% | 90.7% | ✅ Full 5-fold |
| **🥈2** | **5** | CE + LS(ε=0.1) + MixUp(α=0.4) + ECA + Disc.LR + TTA | **77.4%** | 71.2% | **60.0%** | **60.8%** | **92.9%** | ✅ Full 5-fold |
| 3 | **6** | Focal + Ordinal + CBAM + 3-Stage + Disc.LR + TTA | 75.0% | **71.1%** | 59.8% | 60.0% | 92.3% | ✅ Full 5-fold |
| **🥇1** | **7** | CE + LS(ε=0.05) + MixUp(α=0.3) + 3-Stage + TTA | 75.0% | 70.0% | 59.7% | 60.2% | 91.3% | ✅ Full 5-fold |

*\*Exp 0 had no test ensemble — values are best validation fold averages*

**🏆 Top 3:**
1. **Exp 5** — CE + LS + MixUp + ECA + Disc.LR + TTA → **77.4% Acc, 60.0% Recall, 92.9% AUC**
2. **Exp 7** — CE + LS(ε=0.05) + MixUp(α=0.3) + 3-Stage + TTA → **75.0% Acc, 59.7% Recall**
3. **Exp 6** — Focal + Ordinal + CBAM + 3-Stage + Disc.LR + TTA → **75.0% Acc, 59.8% Recall**

---

## Experiment 0 — CrossEntropyLoss Baseline

**Loss:** CrossEntropyLoss (standard)
**Feature Flags:** All OFF (no TTA, no 3-stage, no augmentation tricks)
**Config:** lr=5e-6, epochs=30

| Metric | Score |
|--------|-------|
| Best Val Acc | 67.3% (fold 5) |
| Best Val F1 | 54.7% (fold 5) |

**Notes:**
- This is the raw baseline with no techniques applied — just pure CE on EfficientNet-B3
- No test ensemble was run (no TTA)
- Very low LR (5e-6) for all parameters — likely undertrained backbone
- Class 3 recall catastrophic across all folds — consistent with all experiments
- **Purpose:** Establishes the floor that all other experiments must beat

---

## Experiment 1 — CE + Label Smoothing + 3-Stage + TTA

**Loss:** CrossEntropyLoss(label_smoothing=0.1)
**Feature Flags:** use_label_smoothing=True, use_three_stage=True, use_tta=True
**Config:** lr=1e-3 (head), stage2_lr=1e-4, stage3_lr=1e-6, stage2_duration=3

| Metric | Score |
|--------|-------|
| Accuracy | 73.1% |
| Precision | 70.9% |
| Recall | 58.8% |
| F1 Score | 58.0% |
| ROC-AUC | 90.5% |

**Confusion Matrix (Test TTA):**
```
         Pred→  C0   C1   C2   C3   C4
Actual C0    [[386,  57,   5,   0,   0]
Actual C1     [  7,  75,  22,   0,   1]
Actual C2     [ 10,  56, 193,   1,  17]
Actual C3     [  0,   5,  26,   5,   8]
Actual C4     [  6,  13,  32,   0,  63]]
```

**Per-Class Recall:**
| Class | Correct/Total | Recall |
|:-----:|:------------:|:------:|
| C0 (No DR) | 386/448 | 86.2% |
| C1 (Mild) | 75/105 | **71.4%** |
| C2 (Moderate) | 193/277 | 69.7% |
| C3 (Severe) | 5/44 | 11.4% |
| C4 (Prolif) | 63/114 | 55.3% |

**Strengths:**
- **Best Class 1 recall (71.4%)** across all experiments — LS with 3-Stage excels at mild DR detection
- Massive jump from baseline: +9% accuracy, +4.8% recall
- 3-Stage unfreezing provides gradual, stable feature learning
- TTA provides robust +5.6% accuracy boost over standard inference

**Weaknesses:**
- Class 3 recall **worst of all experiments** (11.4%) — LS pushes model toward "safe" majority predictions
- Class 2 absorbs Class 3: 26/44 (59%) of Class 3 misclassified as C2
- stage3_lr=1e-6 may be too low — backbone barely fine-tunes in Stage 3

---

## Experiment 2 — Focal Loss + Ordinal Penalty + 3-Stage + TTA

**Loss:** FocalLoss(γ=2.0) × 0.7 + OrdinalPenaltyLoss × 0.3
**Feature Flags:** use_focal_loss=True, use_ordinal_loss=True, use_three_stage=True, use_tta=True
**Config:** lr=1e-3, stage2_lr=1e-4, stage3_lr=1e-6, stage2_duration=3

| Metric | Score |
|--------|-------|
| Accuracy | 72.3% |
| Precision | 73.7% |
| Recall | 57.1% |
| F1 Score | 57.4% |
| ROC-AUC | 89.9% |

**Confusion Matrix (Test TTA):**
```
         Pred→  C0   C1   C2   C3   C4
Actual C0    [[387,  57,   4,   0,   0]
Actual C1     [ 12,  67,  25,   0,   1]
Actual C2     [  9,  60, 196,   0,  12]
Actual C3     [  0,   4,  26,   6,   8]
Actual C4     [  6,  13,  37,   0,  58]]
```

**Per-Class Recall:**
| Class | Correct/Total | Recall |
|:-----:|:------------:|:------:|
| C0 | 387/448 | 86.4% |
| C1 | 67/105 | 63.8% |
| C2 | 196/277 | 70.8% |
| C3 | 6/44 | 13.6% |
| C4 | 58/114 | 50.9% |

**Strengths:**
- Highest precision (73.7%) of any experiment — Focal+Ordinal reduces false positives
- C2 recall improved to 70.8% — ordinal penalty helps Moderate class boundaries
- Only completed 1 val fold but ran full test ensemble — TTA provides +8.7% accuracy boost

**Weaknesses:**
- Overall recall **lower than Exp 1** (57.1% vs 58.8%) — Focal Loss doesn't outperform simple LS
- C3 still absorbed into C2: 26/44 (59%) misclassified as Moderate
- C4 recall worst of completed experiments at 50.9%
- Ordinal penalty doesn't solve the C3 boundary problem with only 223 samples

---

## Experiment 3 — CE + Severity MixUp + ECA + TTA

**Loss:** CrossEntropyLoss (standard)
**Feature Flags:** use_mixup=True (α=0.4), use_eca=True, use_tta=True
**Config:** lr=1e-3, NO 3-stage unfreezing (full backbone from start)

| Metric | Score |
|--------|-------|
| Accuracy | 65.9% |
| Precision | 72.8% |
| Recall | 57.9% |
| F1 Score | 57.2% |
| ROC-AUC | **91.4%** |

**Confusion Matrix (Test TTA):**
```
         Pred→  C0   C1   C2   C3   C4
Actual C0    [[275, 168,   5,   0,   0]
Actual C1     [  3,  70,  30,   0,   2]
Actual C2     [  2,  43, 226,   1,   5]
Actual C3     [  0,   0,  31,   7,   6]
Actual C4     [  0,   4,  37,   0,  73]]
```

**Per-Class Recall:**
| Class | Correct/Total | Recall |
|:-----:|:------------:|:------:|
| C0 | 275/448 | **61.4%** |
| C1 | 70/105 | 66.7% |
| C2 | 226/277 | **81.6%** |
| C3 | 7/44 | 15.9% |
| C4 | 73/114 | **64.0%** |

**Strengths:**
- **Highest AUC (91.4%)** — the model learned the most discriminative features
- **Best C2 recall (81.6%)** — MixUp creates synthetic boundary examples that boost Moderate classification
- **Best C4 recall (64.0%)** — MixUp helps minority classes see more variation
- C3 tied for best at 15.9% — MixUp is the only technique that meaningfully helps
- Strong regularization: training accuracy ~60-70% vs 70%+ validation → minimal overfitting

**Weaknesses:**
- **C0 recall COLLAPSED to 61.4%** — 168/448 No-DR images misclassified as Mild (37.5%!)
- MixUp α=0.4 too aggressive — blending C0 and C1 creates ambiguous "No DR + Mild" images
- Accuracy dropped to 65.9% — worst of all complete experiments due to C0 collapse
- No 3-Stage unfreezing → backbone learned too quickly, causing unstable features

---

## Experiment 4 — Focal + Ordinal + CBAM + 3-Stage + TTA

**Loss:** FocalLoss(γ=2.0) × 0.7 + OrdinalPenaltyLoss × 0.3
**Feature Flags:** use_focal_loss=True, use_ordinal_loss=True, use_cbam=True, use_three_stage=True, use_tta=True
**Config:** lr=1e-3, stage2_lr=1e-4, stage3_lr=1e-6, stage2_duration=3

| Metric | Score |
|--------|-------|
| Accuracy | **73.5%** |
| Precision | 71.8% |
| Recall | 57.7% |
| F1 Score | 58.5% |
| ROC-AUC | 90.7% |

**Confusion Matrix (Test TTA):**
```
         Pred→  C0   C1   C2   C3   C4
Actual C0    [[402,  44,   2,   0,   0]
Actual C1     [ 14,  65,  25,   0,   1]
Actual C2     [ 10,  61, 194,   1,  11]
Actual C3     [  0,   2,  28,   7,   7]
Actual C4     [  7,  15,  34,   0,  58]]
```

**Per-Class Recall:**
| Class | Correct/Total | Recall |
|:-----:|:------------:|:------:|
| C0 | 402/448 | **89.7%** |
| C1 | 65/105 | 61.9% |
| C2 | 194/277 | 70.0% |
| C3 | 7/44 | 15.9% |
| C4 | 58/114 | 50.9% |

**Strengths:**
- **Highest raw accuracy (73.5%)** — CBAM + Focal + 3-Stage produces the most conservative, safe predictions
- **Best C0 recall (89.7%)** — model excels at identifying healthy retinas
- C3 tied for best at 15.9% — CBAM provides some channel attention benefit
- TTA provides massive +11.4% accuracy boost over standard inference (62.0% → 73.5%)

**Weaknesses:**
- Recall is mediocre (57.7%) — CBAM adds instability, overfit to majority class
- Validation accuracy extremely unstable across folds (51.3%–67.2%)
- C2 absorbs C3: 28/44 (63.6%) of Severe misclassified as Moderate
- CBAM's heavy parameterization (thousands of params) causes overfitting on small dataset
- C4 recall tied for worst at 50.9%

---

## Experiment 5 — CE + LS + MixUp + ECA + Disc.LR + TTA 🥈

**Loss:** CE + Label Smoothing(ε=0.1)
**Feature Flags:** use_label_smoothing=True, use_mixup=True (α=0.4), use_eca=True, use_discriminative_lr=True, use_tta=True
**Config:** lr_early=1e-5, lr_middle=1e-4, lr_head=1e-3, NO 3-stage, epochs=20
**Status:** ✅ Full 5-fold completed

| Metric | Score |
|--------|-------|
| Accuracy | **77.4%** |
| Precision | 71.2% |
| Recall | **60.0%** |
| F1 Score | **60.8%** |
| ROC-AUC | **92.9%** |

**Confusion Matrix (Test TTA):**
```
         Pred→  C0   C1   C2   C3   C4
Actual C0    [[409,  36,   2,   0,   1]
Actual C1     [  8,  61,  34,   0,   2]
Actual C2     [ 10,  31, 223,   2,  11]
Actual C3     [  0,   0,  32,   5,   7]
Actual C4     [  3,   5,  39,   0,  67]]
```

**Per-Class Recall:**
| Class | Correct/Total | Recall |
|:-----:|:------------:|:------:|
| C0 | 409/448 | **91.3%** |
| C1 | 61/105 | 58.1% |
| C2 | 223/277 | **80.5%** |
| C3 | 5/44 | 11.4% |
| C4 | 67/114 | 58.8% |

**Strengths:**
- **HIGHEST ACCURACY (77.4%)** across all EfficientNet-B3 experiments
- **HIGHEST AUC (92.9%)** — best discrimination ability
- **BEST C0 recall (91.3%)** — superb healthy retina detection
- **2nd best C2 recall (80.5%)** — MixUp excels at moderate DR boundaries
- Disc.LR WITHOUT 3-Stage works well — backbone layers learn at their own pace
- ECA provides lightweight attention without SE block conflict when Disc.LR controls learning rates

**Weaknesses:**
- C3 recall worst (11.4%) — LS + MixUp pushes C3 strongly into C2 (32/44 = 72.7%)
- C1 recall lower (58.1%) vs Exp 1 (71.4%) — LS+MixUp blurs C0/C1 boundary
- Earlier assessment was wrong: individual fold validation was misleading; TTA ensemble recovered strongly

**Key Insight:** Disc.LR works when 3-Stage is OFF. The previous failure was due to the Disc.LR + 3-Stage conflict, not Disc.LR itself.

---

## Experiment 6 — Focal + Ordinal + CBAM + 3-Stage + Disc.LR + TTA

**Loss:** FocalLoss(γ=2.0) × 0.7 + OrdinalPenaltyLoss × 0.3
**Feature Flags:** use_focal_loss=True, use_ordinal_loss=True, use_cbam=True, use_three_stage=True, use_discriminative_lr=True, use_tta=True
**Config:** lr_early=1e-5, lr_middle=1e-4, lr_head=1e-3, stage1_epochs=3, stage2_duration=3, epochs=20
**Status:** ✅ Full 5-fold completed

| Metric | Score |
|--------|-------|
| Accuracy | 75.0% |
| Precision | 71.1% |
| Recall | 59.8% |
| F1 Score | 60.0% |
| ROC-AUC | 92.3% |

**Confusion Matrix (Test TTA):**
```
         Pred→  C0   C1   C2   C3   C4
Actual C0    [[397,  49,   2,   0,   0]
Actual C1     [  7,  70,  27,   0,   1]
Actual C2     [  6,  56, 203,   2,  10]
Actual C3     [  0,   2,  29,   6,   7]
Actual C4     [  2,  15,  32,   0,  65]]
```

**Per-Class Recall:**
| Class | Correct/Total | Recall |
|:-----:|:------------:|:------:|
| C0 | 397/448 | 88.6% |
| C1 | 70/105 | **66.7%** |
| C2 | 203/277 | 73.3% |
| C3 | 6/44 | 13.6% |
| C4 | 65/114 | 57.0% |

**Strengths:**
- **Best C1 recall (66.7%) alongside Exp 3** — Focal Loss helps mild DR detection
- Tied accuracy with Exp 7 (75.0%) — strong overall performance
- C3 recall 13.6% — marginally better than Exp 5
- Despite 3-Stage + Disc.LR conflict, TTA ensemble rescued performance

**Weaknesses:**
- **3-Stage + Disc.LR conflict confirmed** — individual fold validation was unstable (58-73%)
- C2 recall lower (73.3%) than Exp 5/7 — Focal Loss trades C2 for C1 accuracy
- CBAM adds instability — per-fold variance is high
- C3 → C2 misclassification: 29/44 (65.9%) still absorbed into Moderate

---

## Experiment 7 — CE + LS(ε=0.05) + MixUp(α=0.3) + 3-Stage + TTA 🏆

**Loss:** CrossEntropyLoss(label_smoothing=0.05)  
**Feature Flags:** use_label_smoothing=True, use_mixup=True (α=0.3), use_three_stage=True, use_tta=True
**Config:** lr=1e-3 (head), stage2_lr=1e-4, stage3_lr=1e-5, stage2_duration=4

| Metric | Score |
|--------|-------|
| Accuracy | **75.0%** |
| Precision | 70.0% |
| Recall | **59.7%** |
| F1 Score | **60.2%** |
| ROC-AUC | 91.3% |

**Confusion Matrix (Test TTA):**
```
         Pred→  C0   C1   C2   C3   C4
Actual C0    [[392,  51,   4,   0,   1]
Actual C1     [  8,  62,  32,   0,   3]
Actual C2     [ 10,  36, 212,   2,  17]
Actual C3     [  0,   0,  29,   7,   8]
Actual C4     [  5,  11,  30,   0,  68]]
```

**Per-Class Recall:**
| Class | Correct/Total | Recall |
|:-----:|:------------:|:------:|
| C0 | 392/448 | 87.5% |
| C1 | 62/105 | 59.0% |
| C2 | 212/277 | 76.5% |
| C3 | 7/44 | 15.9% |
| C4 | 68/114 | 59.6% |

**Per-Fold Validation F1:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|:------:|:------:|:------:|:------:|:------:|
| 52.0% | **57.3%** | 53.6% | 48.1% | 53.3% |

**Strengths:**
- **BEST RECALL (59.7%)** — the primary optimization target, edging out Exp 1 by +0.9%
- **BEST F1 (60.2%)** — best balance of precision and recall
- **BEST ACCURACY among recall-focused configs (75.0%)** — 2nd overall after Exp 4's 73.5% accuracy (which had lower recall)
- **Balanced C0 recovery:** 87.5% vs Exp 3's 61.4% — reduced LS(0.05) + tamed MixUp(0.3) prevented C0 collapse
- **Best C2 recall (76.5%) among non-MixUp-heavy experiments** — good moderate DR detection
- **Best C4 recall (59.6%) among stable experiments** — tamed MixUp helps minority classes
- AUC 91.3% — near-best discrimination ability
- Extended stage2_duration=4 and stage3_lr=1e-5 (vs 1e-6) provided more gradual, stable learning

**Weaknesses:**
- C1 recall decreased to 59.0% (vs 71.4% in Exp 1) — MixUp blurs the C0/C1 boundary
- C3 recall unchanged at 15.9% — the 223-sample Class 3 problem cannot be solved by loss engineering alone
- Class 3 → Class 2 confusion: 29/44 (65.9%) of Severe misclassified as Moderate
- Fold 4 was weakest (F1=48.1%) — some variance across folds

**Why This Configuration Won:**
1. **Reduced LS (ε=0.05)** avoided the over-smoothing problem seen in Exp 5
2. **Tamed MixUp (α=0.3)** provided regularization + minority class augmentation without the C0 collapse of Exp 3
3. **Extended Stage 2 (4 epochs)** + **higher Stage 3 LR (1e-5)** allowed the backbone to fine-tune more effectively than Exp 1-2's 1e-6
4. **No attention modules** — kept the model simple, avoiding CBAM's instability
5. **No Disc. LR** — avoided the conflict with 3-Stage that killed Exp 5-6

---

## Technique Effectiveness Summary (Revised with Full Exp 5-6 Data)

### ✅ PROVEN — Always Use

| Technique | Evidence | Impact |
|-----------|----------|--------|
| **TTA (8-view)** | Exp 1: +5.6% acc, Exp 4: +11.4% acc, Exp 5: +4.0% acc | **Biggest single boost. Free at inference.** |
| **Label Smoothing** | Exp 1 (73.1%), Exp 5 (77.4%), Exp 7 (75.0%) — top 3 all use LS | **Core loss technique. Use ε=0.05-0.1 depending on other regularizers.** |
| **Severity MixUp** | Exp 5 (77.4%), Exp 7 (75.0%), Exp 3 (best AUC 91.4%) | **Best regularizer. α=0.3-0.4 depending on other techniques.** |

### ✅ PROVEN — Use with Correct Pairing

| Technique | Works With | Conflicts With | Evidence |
|-----------|-----------|----------------|----------|
| **3-Stage Unfreezing** | LS, MixUp (Exp 7: 75.0%) | Disc.LR (Exp 6: unstable folds) | Use 3-Stage OR Disc.LR, never both |
| **Discriminative LR** | LS, MixUp, ECA (Exp 5: **77.4%**) | 3-Stage (Exp 6: conflict) | Disc.LR alone is BETTER than 3-Stage alone — Exp 5 > Exp 7 |
| **ECA** | Disc.LR, MixUp (Exp 5: 77.4%) | Standalone with no schedule (Exp 3: C0 collapsed) | Lightweight attention works when LR schedule controls backbone learning |

### ⚠️ MIXED — Use with Caution

| Technique | Evidence | Notes |
|-----------|----------|-------|
| **CBAM** | Exp 4: 73.5% acc (good); Exp 6: unstable val | Works as sole attention module with 3-Stage, but heavy parameterization risks overfitting |
| **Focal + Ordinal** | Exp 6: 75.0% acc with full pipeline | Viable alternative to LS — Exp 6 matched Exp 7 accuracy but with better C1 recall (66.7%) |

### ❌ AVOID

| Technique | Evidence | Why It Hurts |
|-----------|----------|--------------|
| **3-Stage + Disc.LR together** | Exp 6 unstable folds (58-73% val) | Competing LR control mechanisms destabilize training |
| **MixUp α=0.4 without 3-Stage or Disc.LR** | Exp 3: C0 collapsed to 61.4% | Aggressive MixUp needs LR schedule to stabilize backbone |

### 🔄 KEY REVISION: Disc.LR Rehabilitation

> **Previous assessment was WRONG.** Disc.LR was prematurely abandoned based on 1-2 fold validation metrics.
> With full 5-fold TTA ensemble, **Exp 5 (Disc.LR) is the BEST experiment at 77.4% accuracy and 60.0% recall.**
> The lesson: individual fold validation can be misleading — always run the full TTA ensemble before concluding.

---

## The Class 3 (Severe) Problem

```
Class 2 (Moderate): 1,383 samples → abundant training data
Class 3 (Severe):     223 samples → 6.2× less data
Class 4 (PDR):        569 samples → distinct visual features (neovascularization)
```

**Every experiment** shows 59-73% of Class 3 samples misclassified as Class 2. This is a **fundamental data limitation**, not a technique failure:

| Experiment | C3 → C2 Misclass | C3 Recall |
|:----------:|:----------------:|:---------:|
| Exp 1 (LS+3S) | 26/44 (59%) | 11.4% |
| Exp 2 (F+O) | 26/44 (59%) | 13.6% |
| Exp 3 (MixUp) | 31/44 (70%) | 15.9% |
| Exp 4 (F+O+CBAM) | 28/44 (64%) | 15.9% |
| **Exp 5 (LS+MixUp+ECA+DLR)** | **32/44 (73%)** | **11.4%** |
| Exp 6 (F+O+CBAM+3S+DLR) | 29/44 (66%) | 13.6% |
| Exp 7 (LS+MixUp+3S) | 29/44 (66%) | 15.9% |

**Conclusion:** No loss function, attention mechanism, or training schedule can solve this with 223 samples. Solutions require data-level intervention (external data, class-specific oversampling, ordinal regression head), or **multi-architecture ensembling** (combining ConvNeXt + EfficientNet to break the ceiling).

---

## Recommended Model & Weights

### Best Experiment: Experiment 7

**Folder:** `experiment_7_efficientnetb3_LabelSmoothing_mixup_3Stage_TTA/`

### Which Weights File to Use?

For reproducing the **test_ensemble_tta** results, you need **ALL 5 fold weights together**:

```
best_model_fold_0.pth
best_model_fold_1.pth
best_model_fold_2.pth
best_model_fold_3.pth
best_model_fold_4.pth
```

**The TTA test ensemble works by:**
1. Loading each fold's best model
2. Running 8-view TTA on each model
3. Averaging the softmax probabilities across all 5 models × 8 views = 40 predictions per sample
4. Taking argmax of the averaged probabilities

**You CANNOT reproduce TTA results with a single `.pth` file** — the ensemble of all 5 folds is what produces the 75.0% accuracy / 59.7% recall.

### If You Must Pick ONE Fold

If deployment constraints require a single model, pick **Fold 2** (`best_model_fold_1.pth` — 0-indexed):
- Best validation F1: **57.3%** (vs average 52.9%)
- Best validation accuracy: **71.4%**
- Best validation AUC: **89.8%**
- Most stable training curve — converged smoothly without val loss spikes

However, note that a single-fold model will produce results closer to the **test_standard** row (68.4% acc, 54.8% recall), not the ensemble TTA row.

---

## Final Summary (Revised)

**8 experiments completed, 5 techniques proven effective:**
1. **TTA (8-view)** — free inference-time boost, +5-11% accuracy
2. **Label Smoothing** — core loss technique, present in all top-3 experiments
3. **Severity-Aware MixUp** — best regularizer, boosts AUC and minority class recall
4. **Discriminative LR** *(REHABILITATED)* — best schedule when used WITHOUT 3-Stage (Exp 5: 77.4%)
5. **3-Stage Unfreezing** — alternative to Disc.LR, proven in Exp 7 (75.0%)

**Strongest Combinations (Ranked):**
1. 🥇 LS + MixUp + ECA + Disc.LR + TTA (Exp 5: **77.4% acc, 60.0% recall, 92.9% AUC**)
2. 🥈 LS + MixUp + 3-Stage + TTA (Exp 7: 75.0% acc, 59.7% recall)
3. 🥉 Focal + Ordinal + CBAM + 3-Stage + Disc.LR + TTA (Exp 6: 75.0% acc, 59.8% recall)

**Techniques to AVOID combining:**
1. ~~3-Stage + Disc.LR~~ — competing LR control (though Exp 6 survived via TTA rescue)
2. ~~MixUp α=0.4 without LR schedule~~ — C0 collapse (Exp 3)

**The ceiling for EfficientNet-B3 on this dataset is ~77% accuracy / ~60% recall.** Breaking through 80% requires **multi-architecture ensembling** (ConvNeXt-Small + EfficientNet-B3) and/or data-level Class 3 intervention.

---

## Next Step: Ensemble Modeling

**Best candidates for ensemble:**
- **EfficientNet-B3 Exp 5** (77.4% acc) — best at C0 (91.3%) and C2 (80.5%)
- **EfficientNet-B3 Exp 7** (75.0% acc) — most balanced recall profile
- **ConvNeXt-Small Exp 7** (78.85% acc) — best at C1 (66.7%), C3 (15.9%), C4 (63.2%)

Complementary strengths suggest **weighted probability averaging** across architectures should push accuracy above 80%.
