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
├── baseline_training.ipynb        # CUDA (Windows/Linux)
├── baseline_training_macos.ipynb  # Apple Silicon (MPS)
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

| Device            | Notebook                        |
|-------------------|---------------------------------|
| NVIDIA GPU (CUDA) | `baseline_training.ipynb`       |
| Mac (M1/M2/M3/M4) | `baseline_training_macos.ipynb` |

---

### 🔹 Step 3: Install dependencies

Run Cell 1 in the notebook.

---

### 🔹 Step 4: Hugging Face Login (Optional)

In **Cell 2**, you will see:

```python
from huggingface_hub import notebook_login
notebook_login()
```

👉 Step by step to run this cell:

1. Go to https://huggingface.co/settings/tokens
2. Create an access token (⚠️ **Do not Share it to anyone or push to GitHub**)
3. Paste it into the notebook prompt UI

This part is optional as you're still able to access huggingface API without authentication, just that there's stricter rate limiting. If you have not faced any rate limiting issues can comment out or delete this cell

---

### 🔹 Step 5: Run all cells

The pipeline will:

* Load dataset
* Train model
* Save results into `/experiments/`

---

## 🤝 3. Teammate Model Assignments

Each teammate is assigned a strong model:

| Teammate | Model               |
|----------|---------------------|
| Yi Hui   | Swin Transformer V2 |
| Weng Wai | ConvNeXt V2         |
| Jovan    | EfficientNet        |
| Aloysius | Vision Transformer  |

👉 You are also **free to explore other models** if you believe they perform better.

---

## 🧠 4. How to Change Model (Cell 6)

You need to change:

1. **Import**
2. **Model name in both `AutoImageProcessor.from_pretrained()` and `<Import Name>.from_pretrained()`**

---

### 🔄 Model Conversion Table

| Model          | Import Name                          | <Import Name>.from_pretrained() & AutoImageProcessor.from_pretrained() |
|----------------|--------------------------------------|------------------------------------------------------------------------|
| ViT (baseline) | `ViTForImageClassification`          | `"google/vit-base-patch16-224"`                                        |
| Swin V2        | `Swinv2ForImageClassification`       | `"microsoft/swinv2-base-patch4-window8-256"`                           |
| ConvNeXt V2    | `ConvNextForImageClassification`     | `"facebook/convnextv2-base-1k-224"`                                    |
| EfficientNet   | `EfficientNetForImageClassification` | `"google/efficientnet-b7"`                                             |

---

### 🔧 Example (Swin V2)

```python
from transformers import SwinForImageClassification, AutoImageProcessor

model = SwinForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    num_labels=5
)

processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
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

## 🚀 6. Ways to Improve Performance

Here are recommended strategies:

---

### a) 🔬 Dataset Splitting

* Add Stratified K-Fold using scikit-learn library
* Ensure Fairness dataset distribution between training, validation & testing

---

### b) ⚙️ Hyperparameter Tuning

* Learning rate (e.g., `1e-4`, `3e-5`)
* Batch size (Recommended: `4`, `8`, `16`, `32`)
* Number of epochs (e.g., `10`, `15`, `20`)

---

### c) 🧠 Model Improvements

* Add custom classification head
* Fine-tune deeper layers

---

### d) 🧪 Data Augmentation

* Add Pytorch Data Augmentation (i.e.: `transforms.RandomHorizontalFlip()`, `transforms.ColorJitter()`, `transforms.RandomRotation()`)
* Add CutMix / MixUp
* Implement Stronger rotations / brightness (Adjust based on performance)

---

### e) ⚖️ Imbalance Handling

* Weighted sampling
* Focal loss

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
├── 2026-04-09_swinv2_{strategy}/
├── 2026-04-10_convnext_{strategy}/
├── 2026-04-12_efficientnet_{strategy}/
```

👉 Each PR adds a new experiment folder. This allows us to record and view all past experiments and discuss together the strategy.


## 🏆 9. Model Performance Leaderboard 
| No | Author   | Model              | Strategy Summary                                           | Accuracy | Recall | Date   |
|----|----------|--------------------|------------------------------------------------------------|:--------:|:------:|--------|
| 1. | Aloysius | Vision Transformer | Baseline (Vanila ViT + No Augmentation + Train_Test_Split) |  74.19%  | 58.78% | 4/4/26 |

# 🚀 Let’s Win This Competition

Focus on:

* Improving Recall, Accuracy and F1 score
* Reducing class imbalance errors
* Experimenting smartly

---

Good luck — and build something powerful 💪
