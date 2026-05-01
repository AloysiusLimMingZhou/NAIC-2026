# Troubleshooting

| Problem | What to do |
|---|---|
| Jupyter login fails | Check for spaces in username/password; contact whoever set up your account |
| `git checkout features/ensemble_model` fails | Run `git checkout -b features/ensemble_model origin/features/ensemble_model` |
| Cell 10 `AssertionError: weights must sum to 1.0` | Check that your `per_class_weights` have `convnext[c] + efficientnet[c] == 1.0` for every class |
| Cell 13 prints `SKIP — not found` for all folds | The `weights_dir` path is wrong. Run `ls experiments/` in Terminal and update Cell 10 |
| `RuntimeError: CUDA out of memory` during Cell 13 | Reduce `batch_size` in `run_inference_for_model` from 16 to 8, restart kernel, run again |
| `git push` rejected (non-fast-forward) | Run `git pull origin features/ensemble_model --rebase` then push again |
| Notebook output cells look empty after "Run All" | The kernel may have crashed. Check `Kernel → Restart Kernel and Run All Cells` again |

---
