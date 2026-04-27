# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diabetic Retinopathy (DR) severity classification competition pipeline. 5-class image classification using HuggingFace pretrained vision models fine-tuned on a custom medical imaging dataset. The primary goal is to beat the baseline metrics, especially **Recall** (most important), then Accuracy and F1.

## Repository Structure

- `baseline_training.ipynb` — Standard CUDA training (CrossEntropyLoss, K-Fold, WeightedRandomSampler)
- `baseline_training_advanced.ipynb` — Advanced CUDA training (adds Focal Loss + Three-Stage Fine-Tuning)
- `convnext_training.ipynb` — ConvNeXt-specific training with full optimization stack (CBAM, Ordinal Loss, MixUp, TTA, Snapshot Ensemble)
- `dataset/` — (gitignored) contains `labels.csv` and `image/` directory with ~4,939 DR images
- `experiments/` — each run saves to `exp_{timestamp}/` with `best_model.pth`, `metrics.json`, `config.json`
- `records/` — experiment logs for cross-team comparison, named `YYYY-MM-DD_{model}_{strategy}/`

## Key Notebook Differences

| Notebook | Model | Loss | Fine-Tuning | Extras |
|----------|-------|------|-------------|--------|
| `baseline_training.ipynb` | Any (configurable) | CrossEntropyLoss | Flat LR | K-Fold, CLAHE, WeightedRandomSampler |
| `baseline_training_advanced.ipynb` | Any (configurable) | Focal Loss | Three-Stage Freeze/Unfreeze | + checkpoint recovery |
| `convnext_training.ipynb` | ConvNeXt-Small | Focal + Ordinal (0.7/0.3) | Three-Stage + Discriminative LR | + CBAM, MixUp, TTA, Snapshot Ensemble, Warmup |

All notebooks use CUDA (`torch.device("cuda")`, `autocast(device_type='cuda')`, `GradScaler`).

## Pipeline Architecture (Notebook Cell Flow)

1. **Install deps** — torch, torchvision, transformers, scikit-learn, opencv-python, etc.
2. **Download dataset** — from GCS bucket to local VM disk (reduces read/write cost)
3. **Read CSV** — `labels.csv` with columns: `Image`, `Label` (0-4 severity)
4. **CLAHE preprocessing** — `crop_fundus()` removes black borders, `preprocess_fundus()` applies Green Channel CLAHE
5. **Custom dataset loader** — `MedicalDatasetLoader(Dataset)` with toggle for preprocessing
6. **Device setup** — CUDA
7. **Model loading** — HuggingFace pretrained with `num_labels=5`, `ignore_mismatched_sizes=True`
8. **Data augmentation & splitting** — Stratified 80/20 train+val/test, then K-Fold on train+val
9. **Training loop** — Stratified 5-Fold, WeightedRandomSampler, early stopping, gradient clipping
10. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC (macro, OVR), confusion matrix

## ConvNeXt-Specific Architecture (convnext_training.ipynb)

The ConvNeXt notebook adds these components between the standard pipeline cells:

- **CBAM (Cell 21)**: `ConvNextWithCBAM` wraps the HuggingFace model, inserting Channel+Spatial attention before the classifier. Access backbone via `model.convnext`, attention via `model.cbam`.
- **CombinedLoss (Cell 23)**: `FocalLoss + OrdinalPenaltyLoss` — ordinal penalty weights misclassifications by distance from true class.
- **Severity-aware MixUp (Cell 25)**: Only blends adjacent severity classes (e.g., Class 2 with 1 or 3). Called 50% of the time during training.
- **Feature Flags (Cell 27)**: `FEATURE_FLAGS` dict toggles each technique on/off for ablation experiments.
- **TTA + Snapshot Ensemble (Cell 29)**: 8 augmented views averaged per image, then averaged across all 5 fold models.

When `use_cbam=True`, the model forward pass uses `model(pixel_values)` returning logits directly (not `.logits`). When `use_cbam=False`, use `model(pixel_values=inputs).logits`.

## Model Assignments

| Person   | Model           | HuggingFace Pretrained Name        |
|----------|-----------------|------------------------------------|
| Yi Hui   | ConvNeXt-Small  | `facebook/convnext-small-224`      |
| Aloysius | EfficientNet-B3 | `google/efficientnet-b3`           |

Final goal: ensemble ConvNeXt (0.4) + EfficientNet-B3 (0.6) for architectural diversity.

## Baseline to Beat

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 74.19% |
| Precision | 69.13% |
| Recall    | 58.78% |
| F1 Score  | 59.95% |
| ROC-AUC   | 78.91% |

## Ablation Experiment Order

Run sequentially, recording F1-macro each time for the write-up:
1. ConvNeXt baseline (CrossEntropy, no tricks)
2. + CLAHE + retina cropping
3. + Weighted Focal Loss (gamma=2)
4. + Ordinal Penalty (0.7 Focal + 0.3 Ordinal)
5. + Severity-aware MixUp
6. + Three-stage fine-tuning
7. + CBAM
8. + TTA at inference
9. + Snapshot ensemble across folds

Toggle via `FEATURE_FLAGS` in the training cell.

## Contribution Workflow

1. Create feature branch: `git checkout -b feature/{model}-improvement`
2. Train and iterate — keep only best experiment results in `experiments/`
3. Add experiment record to `records/` as `YYYY-MM-DD_{model}_{strategy}/`
4. Open PR to `main` with model name, metrics, and description of improvements
