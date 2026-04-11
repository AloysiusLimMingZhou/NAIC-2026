# 🧠 Diabetic Retinopathy Classification – Team Training Pipeline

This repository contains the baseline pipeline for training deep learning models on a **Diabetic Retinopathy (DR) severity classification dataset**.

The goal is to:

> 🚀 Improve the baseline model performance using better architectures, preprocessing, and training strategies.

---

## 1. 📁 Project Structure

```
repo/
│
├── dataset/                  # (gitignored) Images + labels.csv
│
├── records/                  # records folder to log everyone's best performing model attempts from time to time
│
├── experiments/              # (gitignored)
│   └── exp_datetime/
│       ├── best_model.pth
│       ├── metrics.json
│       ├── config.json
│
├── baseline_training.ipynb          # Standard training (CUDA)
├── baseline_training_advanced.ipynb # Advanced training — adds Focal Loss + Three-Stage Fine-Tuning (CUDA)
│
├── .gitignore
├── README.md
```

---

## 2. ⚙️ How to Run the Training Script

### 🔹Step 1: Clone the Dependencies
```bash
git clone https://github.com/AloysiusLimMingZhou/NAIC-2026.git
cd NAIC-2026
```

---

### 🔹 Step 2: Choose the correct notebook

| Notebook                           | Implementations                                                                                           |
|------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `baseline_training.ipynb`          | Optimizers, LR Scheduler, EarlyStopping, Stratified 5-Fold, WeightedRandomSampler, CLAHE, Retina Cropping |
| `baseline_training_advanced.ipynb` | Everything above + Weighted Focal Loss (γ=2) + Three-Stage Fine-Tuning + Discriminative LR                |

---

### 🔹 Step 3: Install dependencies

Run Cell 1 in the notebook. Only run once unless new package is needed as it'll be installed inside the VM OS boot disk and will not be removed even after server restart

---

### 🔹 Step 4: Download Dataset from Storage Bucket into Jupyter Server

Run **Cell 2** in the notebook, and it will download the dataset into the VM OS boot disk. This is to reduce Storage Bucket Read/Write cost and latency. 

---

### 🔹 Step 5: Run all cells

The pipeline will:

* Load dataset
* Apply CLAHE preprocessing & retina cropping
* Train model with Stratified 5-Fold + WeightedRandomSampler
* Save results into `/experiments/`

---

## 🤝 3. Teammate Model Assignments

Each teammate is assigned a strong model:

| Teammate                 | Model                                     |
|--------------------------|-------------------------------------------|
| Yi Hui                   | ConvNext-Small                            |
| Aloysius                 | EfficientNet-B3                           |
| Additional (if got time) | EfficientNet-B4, Swin-TransformerV2-Small |

👉 You are also **free to explore other models** if you believe they perform better.

---

## 🧠 4. How to Change Model (Cell 8)

You need to change:

1. **Import**
2. **Model name in both `AutoImageProcessor.from_pretrained()` and `<Import Name>.from_pretrained()`**

---

### 🔄 Model Conversion Table

| Model          | Import Name                          | <Import Name>.from_pretrained() & AutoImageProcessor.from_pretrained() |
|----------------|--------------------------------------|------------------------------------------------------------------------|
| ConvNeXt V2    | `ConvNextForImageClassification`     | `"facebook/convnext-small-224"`                                        |
| EfficientNet   | `EfficientNetForImageClassification` | `"google/efficientnet-b3"`                                             |

---

### 🔧 Example (Swin V2)

```python
from transformers import SwinForImageClassification, AutoImageProcessor

model = SwinForImageClassification.from_pretrained(
    "facebook/convnext-small-224",
    num_labels=5
)

processor = AutoImageProcessor.from_pretrained('facebook/convnext-small-224')
```

**Note:** The model is loaded from **Huggingface**. If you want to explore other models or you think there's typo in the model name, please check out at https://huggingface.co/.

---

## 📊 5. Baseline Performance (You MUST Beat This)

| Metric        | Score  |
|---------------|--------|
| Accuracy      | 74.19% |
| Precision     | 69.13% |
| Recall        | 58.78% |
| F1 Score      | 59.95% |
| ROC_AUC Score | 78.91  |

---

👉 Your improved model should aim to:

> 🎯 **Increase Recall Score (most important metric)**\
> 🎯 **Increase Accuracy Score**\
> 🎯 **Ensure Both Recall & Precision are high (Lead to higher and balance F1 Score)**

---

## 🚀 6. Priority Ranking — What To Actually Implement First

Given GCP L4 GPU, ~4,939 images, competition timeline, here is the honest order of effort vs. reward:

| Priority | Action                                              | Why                                                     | Estimated F1 Gain |
|:--------:|-----------------------------------------------------|---------------------------------------------------------|-------------------|
|    1     | Stratified 5-Fold + WeightedRandomSampler           | Fixes the imbalance problem at source                   | +0.08–0.12        |
|    2     | Green channel CLAHE + retina cropping               | Preprocessing ROI is highest data ROI                   | +0.04–0.06        |
|    3     | Weighted Focal Loss (γ=2)                           | Directly targets Class 3 recall                         | +0.04–0.08        |
|    4     | Three-stage fine-tuning + discriminative LR         | Prevents catastrophic forgetting                        | +0.03–0.05        |
|    5     | EfficientNet-B3 + ConvNeXt-Small ensemble (0.6/0.4) | Architectural diversity drives real gains               | +0.02–0.04        |
|    6     | CBAM attention module                               | Better classification + cleaner Grad-CAM for demo marks | +0.02–0.03        |
|    7     | 8-augmentation TTA at inference                     | Free recall boost, no retraining needed                 | +0.01–0.02        |
|    8     | Snapshot ensemble (best checkpoint per fold)        | Free — you already have the checkpoints                 | +0.01–0.02        |
|    9     | Severity-aware MixUp (±1 class only)                | Genuine minority class variety                          | +0.02–0.03        |
|    10    | Ordinal penalty on loss                             | Clinically meaningful error weighting                   | +0.02–0.04        |

---

## 🧾 7. Contribution Workflow

### 🔹 Step 1: Create your own branch

```bash
git checkout -b feature/swinv2-improvement
```

---

### 🔹 Step 2: Train & Improve Model

* Run the notebook
* Results will be saved in `/experiments/`
* (Important) Only keep the best/final experiment results and delete the rest

---

### 🔹 Step 3: Create your own README

Inside your branch, add:

```
repo/
│
├── model.ipynb
├── README.md
├── experiments/              
│   └── exp_datetime/ (Best Version)
│       ├── metrics.json
│       ├── config.json
```

Your README should include:

* **Model used**
* **Changes made**
* **Final performance**
* **Key insights**

---

### 🔹 Step 4: Open Pull Request (PR)

* Push branch
* Open PR → `main`
* Include:

  * Model name
  * Metrics
  * Improvements

---

## 8. All Experiment Logs Repository Strategy

We will follow a structure similar to **OpenAI parameter-golf style** (https://github.com/openai/parameter-golf):

---

### 📁 Records Folder (All Experiments)

```
records/
│
├── 2026-04-10_convnext_{strategy}/
├── 2026-04-12_efficientnet_{strategy}/
```

👉 Each PR adds a new experiment folder. This allows us to record and view all past experiments and discuss together the strategy.


## 🏆 9. Model Performance Leaderboard 
| No | Author   | Model              | Strategy Summary                                           | Accuracy | Recall | Date   |
|----|----------|--------------------|------------------------------------------------------------|:--------:|:------:|--------|
| 1. | Aloysius | Vision Transformer | Baseline (Vanila ViT + No Augmentation + Train_Test_Split) |  74.19%  | 58.78% | 4/4/26 |

# 🚀 Let's Win This Competition

Focus on:

* Improving Recall, Accuracy and F1 score
* Reducing class imbalance errors
* Experimenting smartly

---

Good luck — and build something powerful 💪
