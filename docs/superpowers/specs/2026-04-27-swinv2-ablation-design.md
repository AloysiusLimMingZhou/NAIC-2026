# SwinV2 Ablation Design — 3-Experiment Plan

**Date:** 2026-04-27
**Model:** SwinV2-Base (`microsoft/swinv2-base-patch4-window8-256`)
**Current best:** 75.61% Acc, 69.80% Recall, 66.86% F1, 94.31% ROC-AUC

## Fixed Across All Experiments
- SwinV2-Base, 256×256 (native resolution)
- Stratified 5-Fold, WeightedRandomSampler
- AdamW, weight_decay=1e-4, CosineAnnealing, eta_min=1e-6
- batch_size=16, epochs=20, patience=10, grad_clip=1.0
- bfloat16 autocast
- TTA (8 views at 256×256)
- RandomErasing(p=0.3)

## Experiment 1 — Ben Graham + Native Resolution + TTA
**Hypothesis:** Running at 224×224 loses pretrained alignment. Ben Graham normalises illumination across cameras.
**Changes from current best:** 224→256, CLAHE→Ben Graham, +TTA, float16→bfloat16
**Loss:** FocalLoss(gamma=1.0) + label_smoothing=0.1 (same as current best)
**Decision gate:** If Ben Graham < CLAHE → revert preprocessing for Exp 2+3

## Experiment 2 — CE+LS(0.03) + Soft-label MixUp
**Hypothesis:** MixUp+Focal conflicts (soft targets break focal modulation). Switch to CE+LS(0.03).
**Changes from Exp 1 best:** Focal+LS(0.1) → CE+LS(0.03), +SoftMixUp(α=0.3), lr 2e-5→1e-4
**Loss:** CrossEntropy + LabelSmoothing(0.03) + SeverityMixUp

## Experiment 3 — ECA + EMA
**Hypothesis:** ECA was +5% F1 on ConvNeXt. EMA (shadow weights) is used in DeiT/BEiT for small datasets.
**Changes from Exp 2 best:** +ECA (on 1024-dim pooled features), +EMA(decay=0.9998, used at TTA inference)
**Loss:** Same as Exp 2

## Notebooks
- `exp1_swin.ipynb` — Exp 1 config
- `exp2_swin.ipynb` — Exp 2 config
- `exp3_swin.ipynb` — Exp 3 config

All based on `SwinV2_Reference.ipynb`, targeted cell changes only.
