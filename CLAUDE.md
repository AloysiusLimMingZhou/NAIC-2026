# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diabetic Retinopathy (DR) severity classification competition pipeline. 5-class image classification using HuggingFace pretrained vision models fine-tuned on a custom medical imaging dataset. The primary goal is to beat the baseline metrics, especially **Recall** (most important), then Accuracy and F1.

## Repository Structure

- `baseline_training.ipynb` — CUDA training notebook (Windows/Linux with NVIDIA GPU)
- `baseline_training_macos.ipynb` — MPS training notebook (Apple Silicon Mac)
- `dataset/` — (gitignored) contains `labels.csv` and `image/` directory with DR images
- `experiments/` — (gitignored) each run saves to `exp_{timestamp}/` with `best_model.pth`, `metrics.json`, `config.json`
- `records/` — experiment logs for cross-team comparison, named `YYYY-MM-DD_{model}_{strategy}/`

## Key Differences Between Notebooks

The two notebooks are nearly identical except:
- **CUDA version**: uses `torch.device("cuda")`, `autocast(device_type='cuda')`, and `GradScaler` for mixed-precision (FP16/FP32)
- **macOS version**: uses `torch.device("mps")`, no autocast/GradScaler (MPS doesn't support AMP the same way)

When modifying training logic, **apply changes to both notebooks** to keep them in sync.

## Pipeline Architecture (Notebook Cell Flow)

1. **Install deps** — torch, torchvision, transformers, scikit-learn, opencv-python, etc.
2. **HuggingFace login** — optional, for rate limit avoidance
3. **CLAHE preprocessing** — `apply_clahe()` converts RGB→LAB, applies CLAHE to L channel, converts back. Applied to training data only.
4. **Custom dataset loader** — `MedicalDatasetLoader(Dataset)` reads from `dataset/labels.csv` (columns: `Image`, `Label`), loads images from `dataset/image/`
5. **Device setup** — CUDA or MPS
6. **Model loading** — HuggingFace pretrained model with `num_labels=5`, `ignore_mismatched_sizes=True`. Uses `AutoImageProcessor` for model-specific normalization mean/std.
7. **Data preprocessing & splitting** — `train_test_split` with stratification (64/16/20 train/val/test). Training uses `train_transform` (with CLAHE), val/test use `val_transform` (no CLAHE, no augmentation).
8. **Hyperparameters** — AdamW optimizer, CosineAnnealingLR scheduler, CrossEntropyLoss with balanced class weights and label smoothing. Early stopping (patience=5).
9. **ExperimentTracker** — logs config, per-epoch metrics, final metrics, and saves model weights to `experiments/exp_{timestamp}/`
10. **Training loop** — standard train/val loop with gradient clipping (max_norm=1.0)
11. **Evaluation** — computes accuracy, precision, recall, F1, ROC-AUC (macro, OVR), confusion matrix on test set

## Model Assignments

| Person   | Model               | HuggingFace Import                       | Pretrained Name                                  |
|----------|---------------------|------------------------------------------|--------------------------------------------------|
| Yi Hui   | Swin Transformer V2 | `Swinv2ForImageClassification`           | `microsoft/swinv2-base-patch4-window8-256`       |
| Weng Wai | ConvNeXt V2         | `ConvNextForImageClassification`         | `facebook/convnextv2-base-1k-224`                |
| Jovan    | EfficientNet        | `EfficientNetForImageClassification`     | `google/efficientnet-b7`                          |
| Aloysius | Vision Transformer  | `ViTForImageClassification`              | `google/vit-base-patch16-224`                     |

To swap models: change the import and the string in both `from_pretrained()` and `AutoImageProcessor.from_pretrained()` in Cell 6.

## Baseline to Beat

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 74.19% |
| Precision | 69.13% |
| Recall    | 58.78% |
| F1 Score  | 59.95% |
| ROC-AUC   | 78.91% |

## Contribution Workflow

1. Create feature branch: `git checkout -b feature/{model}-improvement`
2. Train and iterate — keep only best experiment results in `experiments/`
3. Add experiment record to `records/` as `YYYY-MM-DD_{model}_{strategy}/`
4. Open PR to `main` with model name, metrics, and description of improvements
