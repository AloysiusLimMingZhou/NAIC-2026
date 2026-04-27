"""
Create convnext_best.ipynb from convnext_training.ipynb
Only changes: revert CutMix additions to match Exp 7 config
(CrossEntropyLoss + MixUp + ECA, no CutMix)
"""
import json
import copy

with open('convnext_training.ipynb') as f:
    nb = json.load(f)

best = copy.deepcopy(nb)

# ============================================================
# CELL 25: Remove CutMix functions, keep MixUp only
# ============================================================
cell25_new = '''import torch
import numpy as np

def severity_aware_mixup(images, labels, alpha=0.4):
    """MixUp that only blends samples from adjacent severity classes (+/- 1).
    Returns mixed images and soft label vectors for cross-entropy.

    Args:
        images: (B, C, H, W) tensor
        labels: (B,) tensor of integer class labels 0-4
        alpha: Beta distribution parameter for mixing ratio
    Returns:
        mixed_images: (B, C, H, W) blended images
        labels_a: (B,) original labels
        labels_b: (B,) partner labels
        lam: mixing coefficient
    """
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    # For each sample, find valid partners (adjacent class only)
    partner_indices = torch.zeros(batch_size, dtype=torch.long, device=images.device)

    for i in range(batch_size):
        current_label = labels[i].item()
        # Adjacent classes: current +/- 1, clamped to [0, 4]
        adjacent = []
        if current_label > 0:
            adjacent.append(current_label - 1)
        if current_label < 4:
            adjacent.append(current_label + 1)
        adjacent.append(current_label)  # Allow same-class mixing too

        # Find all samples in the batch that belong to adjacent classes
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=images.device)
        for adj_class in adjacent:
            valid_mask |= (labels == adj_class)

        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        if len(valid_indices) > 0:
            partner_indices[i] = valid_indices[torch.randint(len(valid_indices), (1,))]
        else:
            partner_indices[i] = i  # Fallback: mix with self (no change)

    mixed_images = lam * images + (1 - lam) * images[partner_indices]
    labels_a = labels
    labels_b = labels[partner_indices]

    return mixed_images, labels_a, labels_b, lam

def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    """Compute loss for MixUp: weighted combination of losses for both labels."""
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)'''

best['cells'][25]['source'] = [cell25_new]

# ============================================================
# CELL 27: Change FEATURE_FLAGS and training loop
# ============================================================
cell27_src = ''.join(best['cells'][27]['source'])
lines = cell27_src.split('\n')

# 1. Set use_cutmix to False in FEATURE_FLAGS
new_lines = []
for line in lines:
    if '"use_cutmix": True' in line:
        new_lines.append(line.replace('"use_cutmix": True', '"use_cutmix": False'))
    elif '"CrossEntropyLoss+MixUp+CutMix+ECA"' in line:
        new_lines.append(line.replace('"CrossEntropyLoss+MixUp+CutMix+ECA"', '"CrossEntropyLoss+MixUp+ECA"'))
    elif '"cutmix_alpha"' in line:
        # Remove the cutmix_alpha line entirely
        continue
    else:
        new_lines.append(line)

# 2. Replace the MixUp/CutMix 50/50 training block with MixUp-only
# Find and replace lines 208-232 area
final_lines = []
i = 0
while i < len(new_lines):
    line = new_lines[i]

    # Replace the entire augmentation block
    if '# Apply MixUp or CutMix (50/50 choice)' in line:
        # Get the indentation
        indent = '                '  # 16 spaces (inside for loop + with autocast)
        final_lines.append(f'{indent}# Apply severity-aware MixUp (50% of batches)')
        i += 1
        # Skip until we find the matching else (non-augmented path)
        # We need to rewrite from use_aug through the else branch
        # Find the use_aug line
        while i < len(new_lines):
            if 'use_aug = ' in new_lines[i]:
                final_lines.append(f'{indent}use_aug = FEATURE_FLAGS["use_mixup"] and np.random.random() < 0.5')
                i += 1
                break
            i += 1

        # Skip the old CutMix choice logic until "if use_cutmix_this_batch:" or "mixed_inputs"
        # We want to replace everything from "if use_aug:" to just before the else
        while i < len(new_lines):
            if 'if use_aug:' in new_lines[i]:
                final_lines.append(f'{indent}if use_aug:')
                i += 1
                break
            i += 1

        # Skip old CutMix/MixUp choice block, replace with MixUp-only
        # Skip until we hit the "else:" that pairs with "if use_aug:"
        final_lines.append(f'{indent}    mixed_inputs, labels_a, labels_b, lam = severity_aware_mixup(')
        final_lines.append(f'{indent}        inputs, labels, alpha=HYPERPARAMS_CONFIG["mixup_alpha"]')
        final_lines.append(f'{indent}    )')
        final_lines.append(f'{indent}    if FEATURE_FLAGS["use_cbam"] or FEATURE_FLAGS.get("use_eca", False):')
        final_lines.append(f'{indent}        outputs = model(mixed_inputs)')
        final_lines.append(f'{indent}    else:')
        final_lines.append(f'{indent}        outputs = model(pixel_values=mixed_inputs).logits')
        final_lines.append(f'{indent}    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)')

        # Now skip to the "else:" line in original
        depth = 1  # We're inside "if use_aug:"
        while i < len(new_lines):
            stripped = new_lines[i].strip()
            if stripped == 'else:' and depth == 1:
                # This is the else for use_aug
                final_lines.append(new_lines[i])  # Keep the else line
                i += 1
                break
            # Track if/else depth roughly
            if stripped.startswith('if ') and stripped.endswith(':'):
                depth += 1
            elif stripped == 'else:':
                depth -= 1
            i += 1

        # The else block (non-augmented path) stays the same - keep copying
        continue
    else:
        final_lines.append(line)
        i += 1

best['cells'][27]['source'] = ['\n'.join(final_lines)]

# Verify changes
print("=== Cell 25 check ===")
c25 = ''.join(best['cells'][25]['source'])
print(f"Has severity_aware_mixup: {'severity_aware_mixup' in c25}")
print(f"Has mixup_criterion: {'mixup_criterion' in c25}")
print(f"Has rand_bbox: {'rand_bbox' in c25}")
print(f"Has severity_aware_cutmix: {'severity_aware_cutmix' in c25}")

print("\n=== Cell 27 FEATURE_FLAGS check ===")
c27 = ''.join(best['cells'][27]['source'])
for keyword in ['use_cutmix', 'cutmix_alpha', 'CrossEntropyLoss', 'FEATURE_FLAGS = {']:
    for j, l in enumerate(c27.split('\n')):
        if keyword in l:
            print(f"  L{j}: {l.rstrip()}")

print("\n=== Cell 27 training loop check ===")
for j, l in enumerate(c27.split('\n')):
    if any(k in l for k in ['use_aug', 'severity_aware_mixup', 'severity_aware_cutmix', 'cutmix_this_batch', 'MixUp']):
        print(f"  L{j}: {l.rstrip()}")

print(f"\nTotal cells: {len(best['cells'])}")

with open('convnext_best.ipynb', 'w') as f:
    json.dump(best, f, indent=1)

print("\nDone! convnext_best.ipynb written.")
