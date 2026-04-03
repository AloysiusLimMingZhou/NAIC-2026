# 🧠 Diabetic Retinopathy Classification – Team Training Pipeline

This repository contains the baseline pipeline for training deep learning models on a **Diabetic Retinopathy (DR) severity classification dataset**.

Your goal as a team member is to:

> 🚀 Improve the baseline model performance using better architectures, preprocessing, and training strategies.

---

# 📁 Project Structure

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

# ⚙️ 1. How to Run the Training Script

## 🔹 Step 1: Install dependencies

Run Cell 1 in the notebook.

---

## 🔹 Step 2: Hugging Face Login (IMPORTANT)

In **Cell 2**, you will see:

```python
from huggingface_hub import notebook_login
notebook_login()
```

👉 You MUST:

1. Go to https://huggingface.co/settings/tokens
2. Create an access token
3. Paste it into the notebook prompt

⚠️ Without this, you're using unauthenticated access to huggingface. While that's not an issue, you'll have lower rate limits if you're running it for repeated times. I recommend you all to begin by commenting the notebook_login() function in cell 2, then uncomment it if you've faced any issues.

---

## 🔹 Step 3: Choose the correct notebook

| Device            | Notebook                        |
|-------------------|---------------------------------|
| NVIDIA GPU (CUDA) | `baseline_training.ipynb`       |
| Mac (M1/M2/M3/M4) | `baseline_training_macos.ipynb` |

---

## 🔹 Step 4: Run all cells

The pipeline will:

* Load dataset
* Train model
* Save results into `/experiments/`

---

# 🤝 2. Teammate Model Assignments

Each teammate is assigned a strong model:

| Teammate   | Model               |
|------------|---------------------|
| Teammate 1 | Swin Transformer V2 |
| Teammate 2 | ConvNeXt V2         |
| Teammate 3 | EfficientNet        |

👉 You are also **free to explore other models** if you believe they perform better.

---

# 🧠 3. How to Change Model (Cell 6)

You need to change:

1. **Import**
2. **Model name in `from_pretrained()`**

---

## 🔄 Model Conversion Table

| Model          | Import                               | from_pretrained()                            |
|----------------|--------------------------------------|----------------------------------------------|
| ViT (baseline) | `ViTForImageClassification`          | `"google/vit-base-patch16-224"`              |
| Swin V2        | `Swinv2ForImageClassification`       | `"microsoft/swinv2-base-patch4-window8-256"` |
| ConvNeXt V2    | `ConvNextForImageClassification`     | `"facebook/convnextv2-base-1k-224"`          |
| EfficientNet   | `EfficientNetForImageClassification` | `"google/efficientnet-b7"`                   |

---

## 🔧 Example (Swin V2)

```python
from transformers import SwinForImageClassification

model = SwinForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    num_labels=5
)
```

---

# 📊 4. Baseline Performance (You MUST Beat This)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 82.09% |
| Precision | 70.14% |
| Recall    | 67.99% |
| F1 Score  | 68.97% |

---

👉 Your improved model should aim to:

> 🎯 **Increase F1-score (most important metric)**

---

# 🚀 5. Ways to Improve Performance

Here are recommended strategies:

---

## 🔬 Data Preprocessing

* Add **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
* Improve contrast for medical features

---

## ⚙️ Hyperparameter Tuning

* Learning rate (e.g., `1e-4`, `3e-5`)
* Batch size
* Number of epochs

---

## 🧠 Model Improvements

* Add custom classification head
* Fine-tune deeper layers

---

## 🧪 Data Augmentation

* CutMix / MixUp
* Stronger rotations / brightness

---

## ⚖️ Imbalance Handling

* Weighted sampling
* Focal loss

---

# 🧾 6. Contribution Workflow

## 🔹 Step 1: Create your own branch

```bash
git checkout -b feature/swinv2-improvement
```

---

## 🔹 Step 2: Train & Improve Model

* Run experiments
* Save results in `/experiments/` (auto)

---

## 🔹 Step 3: Create your own README

Inside your branch, add:

```
repo/
│
├── model.ipynb
├── README.md
├── metrics.json (best version)
├── config.json  (best version)
```

Your README should include:

* Model used
* Changes made
* Final performance
* Key insights

---

## 🔹 Step 4: Open Pull Request (PR)

* Push branch
* Open PR → `main`
* Include:

  * Model name
  * Metrics
  * Improvements

---

# 🏆 7. Final Repository Strategy (IMPORTANT)

We will follow a structure similar to **OpenAI parameter-golf style (https://github.com/openai/parameter-golf)**:

---

## 📁 Records Folder (All Experiments)

```
records/
│
├── teammate1_swinv2/
├── teammate2_convnext/
├── teammate3_efficientnet/
```

👉 Each PR adds a new experiment folder

---

## 🏆 Models Folder (BEST ONLY)

```
models/
│
└── model.ipynb
```

👉 Only the **best-performing model** goes here

---

## 🧠 Recommendation (Important)

* Keep ALL experiments (good for learning)
* Only promote the BEST model to `models/`

---

# 🧠 Final Goal

You are not just training models.

You are:

> 💡 **running structured experiments to find the best-performing system**

---

# 🚀 Let’s Win This Competition

Focus on:

* Improving F1-score
* Reducing class imbalance errors
* Experimenting smartly

---

Good luck — and build something powerful 💪
