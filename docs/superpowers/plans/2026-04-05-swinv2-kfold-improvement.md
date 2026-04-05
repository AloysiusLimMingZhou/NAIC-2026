# SwinV2 K-Fold Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ViT baseline with SwinV2 + Stratified K-Fold + Focal Loss + data augmentation + WeightedRandomSampler to beat the baseline recall of 58.78%.

**Architecture:** Minimal K-Fold wrapper approach — keep the existing notebook structure, wrap the training loop in a K-Fold outer loop. Hold out 20% test set first, then 5-fold CV on the remaining 80%. Both macOS and CUDA notebooks updated in sync.

**Tech Stack:** PyTorch, HuggingFace Transformers (Swinv2ForImageClassification), scikit-learn (StratifiedKFold, compute_class_weight), torchvision transforms

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `baseline_training_macos.ipynb` | Modify | Primary notebook — MPS device, no AMP |
| `baseline_training.ipynb` | Modify | CUDA mirror — autocast + GradScaler |

Both notebooks have identical cell structure (11 cells). Changes apply to the same cell numbers in both. Differences are noted per task where CUDA/MPS diverge.

---

### Task 1: Swap Model from ViT to SwinV2 (Cell 6)

**Files:**
- Modify: `baseline_training_macos.ipynb` — Cell 6 (cell id `1ed22a6afb10dcce`)
- Modify: `baseline_training.ipynb` — Cell 6 (cell id `7ddf27d141c26a1a`)

- [ ] **Step 1: Update macOS notebook Cell 6**

Replace the entire Cell 6 code with:

```python
from transformers import Swinv2ForImageClassification, AutoImageProcessor

# Load Microsoft SwinV2 model
model = Swinv2ForImageClassification.from_pretrained(
    'microsoft/swinv2-base-patch4-window8-256',
    num_labels=5,
    ignore_mismatched_sizes=True
).to(device)

processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-base-patch4-window8-256')
```

- [ ] **Step 2: Update Cell 6 markdown header in macOS notebook**

Update the Cell 6 markdown (cell before code) to mention SwinV2 instead of ViT:

```markdown
# 6. Setup Model
In this part, we will load the SwinV2 (Swin Transformer V2) model from HuggingFace. The model is pretrained on ImageNet-1K at 256x256 resolution but we use 224x224 for cross-model consistency.
```

- [ ] **Step 3: Copy the same changes to CUDA notebook Cell 6**

Apply identical code and markdown changes to `baseline_training.ipynb` Cell 6 (cell id `7ddf27d141c26a1a`). The model loading code is the same — `.to(device)` handles CUDA vs MPS automatically.

- [ ] **Step 4: Commit**

```bash
git add baseline_training_macos.ipynb baseline_training.ipynb
git commit -m "feat: swap ViT model to SwinV2-base for image classification"
```

---

### Task 2: Add Data Augmentation to Training Transforms (Cell 7, first half)

**Files:**
- Modify: `baseline_training_macos.ipynb` — Cell 7 (cell id `549096bf43cb5a25`)
- Modify: `baseline_training.ipynb` — Cell 7 (cell id `549096bf43cb5a25`)

- [ ] **Step 1: Update train_transform in macOS notebook Cell 7**

Replace the `train_transform` block with:

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])
```

Keep `val_transform` unchanged.

- [ ] **Step 2: Copy the same train_transform to CUDA notebook Cell 7**

Apply identical `train_transform` change to `baseline_training.ipynb`.

- [ ] **Step 3: Commit**

```bash
git add baseline_training_macos.ipynb baseline_training.ipynb
git commit -m "feat: add data augmentation (flip, rotation, color jitter) to training transforms"
```

---

### Task 3: Replace Data Splitting with Holdout Test + Stratified K-Fold (Cell 7, second half)

**Files:**
- Modify: `baseline_training_macos.ipynb` — Cell 7 (cell id `549096bf43cb5a25`)
- Modify: `baseline_training.ipynb` — Cell 7 (cell id `549096bf43cb5a25`)

This is the biggest change. Cell 7 currently does train/val/test split and creates DataLoaders. We replace the split logic with holdout test + K-Fold setup, but move DataLoader creation into the training loop (Task 6) since it changes per fold.

- [ ] **Step 1: Rewrite macOS notebook Cell 7 splitting logic**

Replace everything in Cell 7 after `val_transform` with:

```python
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd

df = pd.read_csv('dataset/labels.csv')

# Step 1: Hold out 20% as fixed test set (never touched during K-Fold)
train_val_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['Label'],
    random_state=42
)

# Step 2: Setup 5-Fold Stratified K-Fold on the remaining 80%
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Create test dataset and loader (fixed across all folds)
test_dataset = MedicalDatasetLoader(test_df, "dataset/image/", val_transform, use_clahe=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Total samples: {len(df)}")
print(f"Train+Val pool: {len(train_val_df)} (will be split into {N_FOLDS} folds)")
print(f"Test (held out): {len(test_df)}")
```

- [ ] **Step 2: Copy the same changes to CUDA notebook Cell 7**

Apply identical changes to `baseline_training.ipynb` Cell 7.

- [ ] **Step 3: Commit**

```bash
git add baseline_training_macos.ipynb baseline_training.ipynb
git commit -m "feat: replace train_test_split with holdout test + StratifiedKFold setup"
```

---

### Task 4: Add FocalLoss Class (Cell 8)

**Files:**
- Modify: `baseline_training_macos.ipynb` — Cell 8 (cell id `c83caf793f06463b`)
- Modify: `baseline_training.ipynb` — Cell 8 (cell id `c83caf793f06463b`)

Cell 8 currently sets up class weights, optimizer, scheduler, loss, and scaler. We replace it with just the FocalLoss class definition and shared hyperparameter constants. Optimizer/scheduler/criterion creation moves into the K-Fold loop (Task 6) since they must be re-initialized per fold.

- [ ] **Step 1: Rewrite macOS notebook Cell 8**

Replace the entire Cell 8 code with:

```python
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.utils.class_weight import compute_class_weight

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha  # Per-class weights tensor on device
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing via cross_entropy
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        # Get predicted probabilities for focal modulation
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Apply focal modulation: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.mean()

# Shared hyperparameter constants
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
```

- [ ] **Step 2: Rewrite CUDA notebook Cell 8**

Apply the same code to `baseline_training.ipynb` Cell 8, and add the GradScaler import at the end:

```python
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from sklearn.utils.class_weight import compute_class_weight

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.mean()

# Shared hyperparameter constants
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
```

- [ ] **Step 3: Commit**

```bash
git add baseline_training_macos.ipynb baseline_training.ipynb
git commit -m "feat: add FocalLoss class with gamma=2.0 and label smoothing"
```

---

### Task 5: Update ExperimentTracker for K-Fold (Cell 9)

**Files:**
- Modify: `baseline_training_macos.ipynb` — Cell 9 (cell id `99db575e89f1376e`)
- Modify: `baseline_training.ipynb` — Cell 9 (cell id `99db575e89f1376e`)

Add methods to log per-fold metrics and compute aggregated (mean +/- std) results.

- [ ] **Step 1: Update ExperimentTracker in macOS notebook Cell 9**

Replace the entire Cell 9 code with:

```python
import json
from datetime import datetime
import os

class ExperimentTracker:
    def __init__(self, base_dir="experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
        os.makedirs(self.exp_dir)

        self.epoch_metrics = []
        self.fold_metrics = []
        self.final_metrics = {}
        self.config = {}

    # ---------------- CONFIG ----------------
    def log_config(self, config):
        self.config = config
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    # ---------------- PER EPOCH ----------------
    def log_epoch(self, fold, epoch, train_loss, val_loss, train_acc, val_acc):
        self.epoch_metrics.append({
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

    # ---------------- PER FOLD ----------------
    def log_fold_metrics(self, fold, acc, prec, rec, f1, roc_auc):
        self.fold_metrics.append({
            "fold": fold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc_score": roc_auc
        })

    # ---------------- AGGREGATED K-FOLD ----------------
    def log_aggregated_metrics(self):
        if not self.fold_metrics:
            return {}
        metrics_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc_score"]
        aggregated = {}
        for key in metrics_keys:
            values = [fm[key] for fm in self.fold_metrics]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_fold": values
            }
        self.final_metrics["kfold_aggregated"] = aggregated
        return aggregated

    # ---------------- FINAL METRICS ----------------
    def log_final_metrics(self, split, acc, prec, rec, f1, roc_auc, cm):
        self.final_metrics[split] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc_score": roc_auc,
            "confusion_matrix": cm.tolist()
        }

    # ---------------- SAVE ----------------
    def save_all(self):
        with open(os.path.join(self.exp_dir, "metrics.json"), "w") as f:
            json.dump({
                "config": self.config,
                "epoch_metrics": self.epoch_metrics,
                "fold_metrics": self.fold_metrics,
                "final_metrics": self.final_metrics
            }, f, indent=4)

    def save_model(self, model, name="best_model.pth"):
        torch.save(model.state_dict(), os.path.join(self.exp_dir, name))
```

- [ ] **Step 2: Copy the same ExperimentTracker to CUDA notebook Cell 9**

Apply identical code to `baseline_training.ipynb` Cell 9.

- [ ] **Step 3: Commit**

```bash
git add baseline_training_macos.ipynb baseline_training.ipynb
git commit -m "feat: update ExperimentTracker with per-fold and aggregated K-Fold metrics"
```

---

### Task 6: Rewrite Training Loop with K-Fold (Cell 10) — macOS Version

**Files:**
- Modify: `baseline_training_macos.ipynb` — Cell 10 (cell id `f06e6839eb586cf2`)

This is the core change. The entire Cell 10 is replaced with the K-Fold training loop. This is the **macOS version** (no autocast, no GradScaler).

- [ ] **Step 1: Replace macOS notebook Cell 10 with K-Fold training loop**

Replace the entire Cell 10 code with:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

CONFIG = {
    "model": "microsoft/swinv2-base-patch4-window8-256",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "loss": "FocalLoss",
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,
    "k_folds": N_FOLDS,
    "augmentation": "HFlip+VFlip+Rotation15+ColorJitter",
    "sampler": "WeightedRandomSampler"
}

tracker = ExperimentTracker()
tracker.log_config(CONFIG)

# Track best model across all folds
global_best_val_loss = float('inf')
best_fold = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['Label'])):
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{N_FOLDS}")
    print(f"{'='*60}")

    # --- Split data for this fold ---
    fold_train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    fold_val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # --- Create datasets ---
    fold_train_dataset = MedicalDatasetLoader(fold_train_df, "dataset/image/", train_transform, use_clahe=True)
    fold_val_dataset = MedicalDatasetLoader(fold_val_df, "dataset/image/", val_transform, use_clahe=False)

    # --- WeightedRandomSampler for this fold's training data ---
    fold_labels = fold_train_df['Label'].values
    class_counts = np.bincount(fold_labels)
    sample_weights = 1.0 / class_counts[fold_labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # --- DataLoaders ---
    fold_train_loader = DataLoader(fold_train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    fold_val_loader = DataLoader(fold_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Re-initialize model fresh for each fold ---
    model = Swinv2ForImageClassification.from_pretrained(
        'microsoft/swinv2-base-patch4-window8-256',
        num_labels=5,
        ignore_mismatched_sizes=True
    ).to(device)

    # --- Compute class weights for this fold ---
    class_weights = compute_class_weight('balanced', classes=np.unique(fold_train_df['Label']), y=fold_train_df['Label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # --- Loss, optimizer, scheduler ---
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # --- Early stopping state ---
    best_val_loss = float('inf')
    epochs_no_improvement = 0

    for epoch in range(NUM_EPOCHS):
        # ---------------- TRAINING ----------------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in fold_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(pixel_values=inputs)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in fold_val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(pixel_values=inputs)
                loss = criterion(outputs.logits, labels)

                probs = F.softmax(outputs.logits, dim=1)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.logits, 1)

                val_total += inputs.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

        # ---------------- EPOCH METRICS ----------------
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step()

        tracker.log_epoch(fold + 1, epoch, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc)

        # --- Early stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improvement = 0
            # Save if this is the best model across ALL folds
            if epoch_val_loss < global_best_val_loss:
                global_best_val_loss = epoch_val_loss
                best_fold = fold + 1
                tracker.save_model(model)
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    # --- End-of-fold validation metrics ---
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro')
    try:
        roc_auc = roc_auc_score(np.array(all_labels), np.array(all_probs), multi_class='ovr')
    except ValueError:
        roc_auc = 0.0

    tracker.log_fold_metrics(fold + 1, acc, prec, rec, f1, roc_auc)
    print(f"\n  Fold {fold+1} Val — Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")

# --- Aggregated K-Fold results ---
aggregated = tracker.log_aggregated_metrics()
print(f"\n{'='*60}")
print(f"K-FOLD AGGREGATED RESULTS (best model from fold {best_fold})")
print(f"{'='*60}")
for metric, vals in aggregated.items():
    print(f"  {metric}: {vals['mean']*100:.2f}% (+/- {vals['std']*100:.2f}%)")

tracker.save_all()
```

- [ ] **Step 2: Commit**

```bash
git add baseline_training_macos.ipynb
git commit -m "feat: rewrite macOS training loop with 5-fold K-Fold, FocalLoss, WeightedRandomSampler"
```

---

### Task 7: Rewrite Training Loop with K-Fold (Cell 10) — CUDA Version

**Files:**
- Modify: `baseline_training.ipynb` — Cell 10 (cell id `f06e6839eb586cf2`)

Same logic as Task 6 but with CUDA-specific AMP (autocast + GradScaler).

- [ ] **Step 1: Replace CUDA notebook Cell 10 with K-Fold training loop**

Replace the entire Cell 10 code with the same code as Task 6 Step 1, but with these differences in the training block inside the epoch loop:

```python
        # ---------------- TRAINING (CUDA with AMP) ----------------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in fold_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(pixel_values=inputs)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
```

And the validation block:

```python
        with torch.no_grad():
            for inputs, labels in fold_val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast(device_type='cuda'):
                    outputs = model(pixel_values=inputs)
                    loss = criterion(outputs.logits, labels)

                probs = F.softmax(outputs.logits, dim=1)
                # ... rest identical to macOS version
```

Also add at the top of Cell 10 (before CONFIG):

```python
from torch.amp import autocast
```

And inside the fold loop, after creating the optimizer, add:

```python
    scaler = GradScaler()
```

Everything else (CONFIG, K-Fold loop structure, early stopping, metrics, aggregation) is identical to the macOS version in Task 6.

- [ ] **Step 2: Commit**

```bash
git add baseline_training.ipynb
git commit -m "feat: rewrite CUDA training loop with 5-fold K-Fold, FocalLoss, AMP support"
```

---

### Task 8: Update Test Evaluation to Use Best Fold Model (Cell 11)

**Files:**
- Modify: `baseline_training_macos.ipynb` — Cell 11 (cell id `90648d7592a9f2f5`)
- Modify: `baseline_training.ipynb` — Cell 11 (cell id `90648d7592a9f2f5`)

The evaluate function stays mostly the same, but we need to load the best fold's model before evaluating on the held-out test set.

- [ ] **Step 1: Update macOS notebook Cell 11**

Replace the entire Cell 11 code with:

```python
def evaluate_and_log(model, test_loader, device, tracker):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(pixel_values=inputs)

            probs = F.softmax(outputs.logits, dim=1)
            _, predicted = torch.max(outputs.logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    try:
        roc_auc = roc_auc_score(
            np.array(all_labels),
            np.array(all_probs),
            multi_class='ovr'
        )
    except ValueError:
        roc_auc = 0.0

    tracker.log_final_metrics("test", acc, prec, rec, f1, roc_auc, cm)
    tracker.save_all()

    print(f"\nTEST RESULTS (Best model from Fold {best_fold})")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")

    return acc, prec, rec, f1, roc_auc, cm

# Load best model from K-Fold training and evaluate on held-out test set
best_model = Swinv2ForImageClassification.from_pretrained(
    'microsoft/swinv2-base-patch4-window8-256',
    num_labels=5,
    ignore_mismatched_sizes=True
).to(device)

best_model.load_state_dict(torch.load(os.path.join(tracker.exp_dir, "best_model.pth"), map_location=device))

evaluate_and_log(best_model, test_loader, device, tracker)
```

- [ ] **Step 2: Copy the same changes to CUDA notebook Cell 11**

Apply identical code to `baseline_training.ipynb` Cell 11. The code is device-agnostic (`map_location=device` handles both CUDA and MPS).

- [ ] **Step 3: Commit**

```bash
git add baseline_training_macos.ipynb baseline_training.ipynb
git commit -m "feat: update test evaluation to load best K-Fold model and report final metrics"
```

---

### Task 9: Final Review and Cleanup

**Files:**
- Review: `baseline_training_macos.ipynb` — all cells
- Review: `baseline_training.ipynb` — all cells

- [ ] **Step 1: Verify import consistency across both notebooks**

Check that all necessary imports are present in the correct cells:
- Cell 7: `StratifiedKFold`, `WeightedRandomSampler` imported
- Cell 8: `FocalLoss` class defined, `F` imported from `torch.nn.functional`
- Cell 10: `F` available (imported in Cell 8), `Swinv2ForImageClassification` available (imported in Cell 6)
- Cell 11: `os` available (imported in Cell 9), `Swinv2ForImageClassification` available (imported in Cell 6)

- [ ] **Step 2: Verify CUDA notebook has autocast/GradScaler in all correct places**

Check that `baseline_training.ipynb`:
- Cell 8: imports `GradScaler` from `torch.amp`
- Cell 10: imports `autocast` from `torch.amp`, creates `scaler = GradScaler()` inside fold loop, wraps forward pass in `autocast`, uses `scaler.scale/unscale_/step/update`

- [ ] **Step 3: Verify macOS notebook does NOT have autocast/GradScaler**

Check that `baseline_training_macos.ipynb`:
- Cell 8: does NOT import `GradScaler`
- Cell 10: does NOT use `autocast` or `scaler`

- [ ] **Step 4: Commit final cleanup if any changes needed**

```bash
git add baseline_training_macos.ipynb baseline_training.ipynb
git commit -m "chore: verify import consistency and CUDA/MPS parity across notebooks"
```

---

## Summary of Cell Changes

| Cell | Before | After |
|------|--------|-------|
| 1-5 | Unchanged | Unchanged |
| 6 | ViT model + processor | SwinV2 model + processor |
| 7 | train_test_split → train/val/test + DataLoaders | Augmented train_transform + holdout test + StratifiedKFold setup |
| 8 | Class weights, optimizer, scheduler, CE loss, scaler | FocalLoss class + hyperparameter constants |
| 9 | ExperimentTracker (basic) | ExperimentTracker + fold metrics + aggregation |
| 10 | Single training loop | K-Fold outer loop + WeightedRandomSampler + FocalLoss + per-fold tracking |
| 11 | evaluate_and_log + call | evaluate_and_log + load best fold model + call |
