# Ensemble Strategy Experiments Guide

**Goal:** Combine ConvNeXt-Small (78.85% acc) + EfficientNet-B3 (77.4% acc) to break 80% accuracy.

Each experiment below maps to a specific toggle configuration in `ensemble_ultimate.ipynb`.

---

## Models Used

| Model | Best Experiment | Accuracy | Recall | Weights |
|-------|:--------------:|:--------:|:------:|---------|
| **ConvNeXt-Small** | Exp 7 (CE+MixUp+ECA) | 78.85% | 63.58% | `PATH/TO/CONVNEXT/best_model_fold_{0-4}.pth` |
| **EfficientNet-B3** | Exp 5 (CE+LS+MixUp+ECA+DLR) | 77.4% | 60.0% | `experiments/exp_5_.../best_model_fold_{0-4}.pth` |

> **Note:** Update `MODEL_PATHS` in Cell 2 of the notebook with the actual weight paths before running.

---

## Experiment 1: Simple Probability Averaging (Baseline)

**Description:** Equal-weight average of softmax probabilities from both models. This is the simplest ensemble — establishes the ensemble baseline.

**Formula:** `p_ensemble = 0.5 × p_ConvNeXt + 0.5 × p_EfficientNet`

```python
ENSEMBLE_FLAGS = {
    "use_tta": True,
    "strategy": "simple_avg",
}
```

**Expected outcome:** Should beat both individual models. Conservative estimate: 79-80% accuracy.

---

## Experiment 2: Weighted Probability Averaging

**Description:** Weight ConvNeXt higher since it has better overall accuracy and per-class recall. Weights are tunable.

**Formula:** `p_ensemble = 0.55 × p_ConvNeXt + 0.45 × p_EfficientNet`

```python
ENSEMBLE_FLAGS = {
    "use_tta": True,
    "strategy": "weighted_avg",
    "convnext_weight": 0.55,
    "efficientnet_weight": 0.45,
}
```

**Tuning tips:**
- If ConvNeXt dominates accuracy → increase to 0.60/0.40
- If EfficientNet's C0 recall helps → try 0.50/0.50
- Try 0.60/0.40 and 0.65/0.35 as follow-ups

---

## Experiment 3: Per-Class Weighted Averaging

**Description:** Different weights for each class, exploiting each model's per-class strengths. This is the most targeted strategy.

**Formula:** `p_ensemble[c] = w_convnext[c] × p_ConvNeXt[c] + w_effnet[c] × p_EfficientNet[c]`

```python
ENSEMBLE_FLAGS = {
    "use_tta": True,
    "strategy": "per_class_weighted",
    "per_class_weights": {
        # Based on per-class recall analysis:
        # ConvNeXt: C0=89.5%, C1=66.7%, C2=82.7%, C3=15.9%, C4=63.2%
        # EffNet:   C0=91.3%, C1=58.1%, C2=80.5%, C3=11.4%, C4=58.8%
        "convnext":     [0.40, 0.65, 0.50, 0.60, 0.55],
        "efficientnet": [0.60, 0.35, 0.50, 0.40, 0.45],
    },
}
```

**Weight rationale:**
| Class | ConvNeXt | EfficientNet | Why |
|:-----:|:--------:|:------------:|-----|
| C0 | 0.40 | **0.60** | EffNet has better C0 recall (91.3% vs 89.5%) |
| C1 | **0.65** | 0.35 | ConvNeXt dominates C1 (66.7% vs 58.1%) |
| C2 | 0.50 | 0.50 | Similar C2 recall — average both |
| C3 | **0.60** | 0.40 | ConvNeXt slightly better C3 (15.9% vs 11.4%) |
| C4 | **0.55** | 0.45 | ConvNeXt has better C4 (63.2% vs 58.8%) |

---

## Experiment 4: Majority Voting (Hard Voting)

**Description:** Each model × fold produces a hard prediction (argmax). Majority vote across all 10 voters (2 models × 5 folds). Ties broken by ConvNeXt (higher accuracy).

```python
ENSEMBLE_FLAGS = {
    "use_tta": True,
    "strategy": "majority_vote",
}
```

**Expected outcome:** More conservative than probability averaging — may increase precision but lower recall for minority classes. Good for maximizing accuracy.

---

## Experiment 5: Rank-Based Fusion (Borda Count)

**Description:** Each model ranks the 5 classes by probability. Scores are summed across models. The class with the highest total score wins. Robust to probability miscalibration between models.

**Formula:** `score[c] = Σ (K - rank_model[c])` where K=5 classes

```python
ENSEMBLE_FLAGS = {
    "use_tta": True,
    "strategy": "rank_fusion",
}
```

**Expected outcome:** Most robust to differences in probability calibration between ConvNeXt and EfficientNet. May outperform weighted averaging when raw probabilities are poorly calibrated.

---

## Recommended Experiment Order

| Order | Strategy | Why First |
|:-----:|----------|-----------|
| 1 | **Simple Averaging** | Establishes the ensemble baseline — cheapest to evaluate |
| 2 | **Weighted Averaging** (0.55/0.45) | Quick test of whether ConvNeXt should be weighted higher |
| 3 | **Per-Class Weighted** | Most targeted — exploits per-class complementarity |
| 4 | **Majority Voting** | Different aggregation paradigm — may surprise |
| 5 | **Rank Fusion** | Most robust fallback if calibration differs |

---

## After Running All 5 Experiments

The notebook's Cell 19 (`Strategy Comparison`) can be run to produce a side-by-side comparison table of all strategies you've tested. Toggle `"strategy": "all"` to automatically run and compare all 5 strategies in a single cell.

**What to look for:**
1. Which strategy has the **highest accuracy**? → primary metric
2. Which strategy has the **best Class 3 recall**? → the bottleneck class
3. Which strategy has the **best macro recall**? → the balanced metric

**If no strategy beats 80%:** Consider adding the EfficientNet-B3 Exp 7 weights as a third model in the ensemble for additional diversity.
