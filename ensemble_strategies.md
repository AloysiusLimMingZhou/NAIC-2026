# Ensemble Strategy Experiments Guide

**Goal:** Combine ConvNeXt-Small (78.85% acc) + EfficientNet-B3 (77.4% acc) to break 80% accuracy.

Each experiment below maps to a specific toggle configuration in `ensemble_ultimate_fixed.ipynb`. Change **Cell 10** to switch between strategies.

---

## Models Used

| Model | Best Experiment | Accuracy | Recall | Per-Fold Weights |
|-------|:--------------:|:--------:|:------:|---------|
| **ConvNeXt-Small** | Exp 7 (CE+MixUp+ECA) | 78.85% | 63.58% | `<convnext-exp7-dir>/best_model_fold_{0-4}.pth` |
| **EfficientNet-B3** | Exp 5 (CE+LS+MixUp+ECA+DLR) | 77.40% | 60.00% | `<effnet-exp5-dir>/best_model_fold_{0-4}.pth` |

> **Note:** Update `MODEL_PATHS` in **Cell 10** of the notebook (not Cell 2 — that's the imports cell) with the actual weight directory names before running. Cell 10 validates your configuration and will throw an `AssertionError` if flags are inconsistent.

**What gets ensembled**: **10 weight files total** per run — 5 folds of ConvNeXt × 5 folds of EfficientNet. Each fold additionally runs 8 TTA views, so every test image goes through **80 forward passes** before the ensemble strategy is applied.

---

## Experiment 1: Simple Probability Averaging (Baseline)

**Description:** Equal-weight average of per-sample softmax probabilities from both models (each already averaged across its 5 folds and 8 TTA views). The simplest ensemble — establishes the floor every other strategy must beat.

**Formula:** `p_ensemble = 0.5 × p_ConvNeXt + 0.5 × p_EfficientNet`

```python
ENSEMBLE_FLAGS = {
    'strategy':         'simple_avg',
    'use_tta':          True,
    'use_efficientnet': True,
    'use_convnext':     True,
    'effnet_attention':   'eca',
    'convnext_attention': 'eca',
}
```

**Expected outcome:** Should beat both individual models. Conservative estimate: 79–80% accuracy.

---

## Experiment 2: Weighted Probability Averaging

**Description:** Weight ConvNeXt higher since it has better overall accuracy and per-class recall. Weights must sum to 1.0 (the notebook's Cell 10 asserts this).

**Formula:** `p_ensemble = w_convnext × p_ConvNeXt + w_effnet × p_EfficientNet`

```python
ENSEMBLE_FLAGS = {
    'strategy':            'weighted_avg',
    'use_tta':             True,
    'use_efficientnet':    True,
    'use_convnext':        True,
    'effnet_attention':    'eca',
    'convnext_attention':  'eca',
    'convnext_weight':     0.55,
    'efficientnet_weight': 0.45,
}
```

**Tuning tips:**
- If ConvNeXt dominates accuracy → try 0.60/0.40 or 0.65/0.35
- If EfficientNet's C0 recall is critical → try 0.50/0.50
- Follow-up sweep: 0.60/0.40 and 0.65/0.35

---

## Experiment 3: Per-Class Weighted Averaging

**Description:** Different weights for each class, exploiting each model's per-class strengths. The most targeted strategy. Per-class weights must sum to 1.0 **for each class** (Cell 10 asserts this).

**Formula:** `p_ensemble[c] = w_convnext[c] × p_ConvNeXt[c] + w_effnet[c] × p_EfficientNet[c]`

```python
ENSEMBLE_FLAGS = {
    'strategy':           'per_class_weighted',
    'use_tta':            True,
    'use_efficientnet':   True,
    'use_convnext':       True,
    'effnet_attention':   'eca',
    'convnext_attention': 'eca',
    'per_class_weights': {
        # Based on per-class recall analysis:
        # ConvNeXt: C0=89.5%, C1=66.7%, C2=82.7%, C3=15.9%, C4=63.2%
        # EffNet:   C0=91.3%, C1=58.1%, C2=80.5%, C3=11.4%, C4=58.8%
        'convnext':     [0.40, 0.65, 0.50, 0.60, 0.55],
        'efficientnet': [0.60, 0.35, 0.50, 0.40, 0.45],
    },
}
```

**Weight rationale:**

| Class | ConvNeXt | EfficientNet | Why |
|:-----:|:--------:|:------------:|-----|
| C0 | 0.40 | **0.60** | EffNet has better C0 recall (91.3% vs 89.5%) |
| C1 | **0.65** | 0.35 | ConvNeXt dominates C1 (66.7% vs 58.1%) |
| C2 | 0.50 | 0.50 | Similar C2 recall — equal weight |
| C3 | **0.60** | 0.40 | ConvNeXt slightly better C3 (15.9% vs 11.4%) |
| C4 | **0.55** | 0.45 | ConvNeXt has better C4 (63.2% vs 58.8%) |

**Because weights sum to 1.0 per class**, the output row still sums to 1.0 — no renormalization needed. This makes per-class weighted a proper probability blend, not a heuristic score.

---

## Experiment 4: Majority Voting (Hard Voting)

**Description:** Each model produces one hard prediction (argmax of its averaged probs). If both models agree → that class wins. If they disagree → **ConvNeXt breaks the tie** (since ConvNeXt has slightly higher overall accuracy). The notebook blends 50% one-hot vote + 50% soft probabilities so ROC-AUC remains meaningful.

```python
ENSEMBLE_FLAGS = {
    'strategy':           'majority_vote',
    'use_tta':            True,
    'use_efficientnet':   True,
    'use_convnext':       True,
    'effnet_attention':   'eca',
    'convnext_attention': 'eca',
}
```

**Expected outcome:** More conservative than probability averaging — may increase precision but can lower recall for minority classes. Best when you want to maximize accuracy at the cost of Class 3 recall.

> **Note**: "Voting across 10 folds" doesn't happen at the vote level — each model's 5 folds are first averaged into a single probability matrix, then the argmax of that matrix is the model's "vote". This avoids one model with noisy early folds dominating by sheer count.

---

## Experiment 5: Rank-Based Fusion (Borda Count)

**Description:** Each model ranks the 5 classes from lowest to highest probability. Ranks are summed across both models. The class with the highest summed rank wins. **Robust to probability miscalibration** — useful when one model is consistently over-confident or under-confident in its softmax outputs.

**Formula:**
```
rank_model[c]  = 0..(K-1)  where K is highest-probability class
score[c]       = rank_convnext[c] + rank_effnet[c]
winner         = argmax(score)
```

```python
ENSEMBLE_FLAGS = {
    'strategy':           'rank_fusion',
    'use_tta':            True,
    'use_efficientnet':   True,
    'use_convnext':       True,
    'effnet_attention':   'eca',
    'convnext_attention': 'eca',
}
```

**Expected outcome:** Most robust to differences in probability calibration between ConvNeXt and EfficientNet. May outperform weighted averaging when raw probabilities are poorly calibrated (e.g., when one model was trained with Label Smoothing and the other wasn't, which is exactly our case).

---

## Recommended Experiment Order

Run with `strategy: 'all'` for a one-shot comparison. If you want to step through manually:

| Order | Strategy | Rationale |
|:-----:|----------|-----------|
| 1 | **Simple Averaging** | Establishes the ensemble floor. Cheapest interpretation. |
| 2 | **Weighted Averaging** (0.55/0.45) | Tests whether ConvNeXt should be weighted higher globally. |
| 3 | **Per-Class Weighted** | Most targeted — exploits per-class complementarity. Usually wins. |
| 4 | **Majority Voting** | Different aggregation paradigm — may surprise on accuracy. |
| 5 | **Rank Fusion** | Most robust fallback if calibration between models differs significantly. |

---

## Running All 5 Strategies at Once

The notebook has **first-class support** for running all five strategies in a single pass:

```python
ENSEMBLE_FLAGS = {
    'strategy': 'all',   # ← runs every strategy, prints comparison table
    'use_tta':  True,
    # ... rest unchanged
}
```

**Cell 16** (Strategy Comparison Summary) then produces a side-by-side comparison table and identifies the best strategy per metric. This is the recommended approach since the expensive part (running inference on all 10 weight files with TTA) only happens once — swapping strategies is essentially free compute after inference is done.

**What to look for in Cell 16's output:**
1. Which strategy has the **highest accuracy**? → primary leaderboard metric
2. Which strategy has the **best Class 3 recall**? → the bottleneck class (target: >16% to beat all solo models)
3. Which strategy has the **best macro recall**? → the class-balanced metric

---

## If No Strategy Beats 80%

**Three fallback options**, in order of cost:

1. **Re-tune global weights** — the 0.55/0.45 default is educated but not optimized. Try 0.50/0.50, 0.60/0.40, 0.65/0.35 and keep whichever gives highest validation accuracy.

2. **Re-tune per-class weights** — if Class 3 recall is the holdup, push ConvNeXt's C3 weight higher (e.g., 0.70 convnext / 0.30 effnet). Watch that the class weights still sum to 1.0.

3. **Add a third model** — EfficientNet-B3 Exp 7 (75.0% acc, balanced per-class) provides architectural diversity. Cell 10 currently supports two models; extending to three means adding a third entry to `MODEL_PATHS` and generalizing the strategy functions — see the section below.

### Architectural note on adding a 3rd model

The current strategies are 2-model-aware. To support 3+ models, the cleanest refactor is:

- Replace `effnet_probs`, `convnext_probs` with a list `model_probs: list[Tensor]`
- Replace scalar weights with `model_weights: list[float]` (still summing to 1.0)
- Per-class weights become `Tensor(num_models, num_classes)` with columns summing to 1.0

The math generalizes cleanly — just more verbose assertions in Cell 10.

---

## Output Files

After a successful run, Cell 17 writes:
```
ensemble_results/ensemble_<timestamp>.json
```

Contents:
- Full `ENSEMBLE_FLAGS` config (for reproducibility)
- Full `MODEL_PATHS` (so anyone can re-run with the same weights)
- Metrics for each strategy tested
- Confusion matrix for each strategy
- Best-per-metric summary

Commit this JSON to `features/ensemble_model` branch so the team has a permanent record of every run.
