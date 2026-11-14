# Model checkpoints (optional in git)

Full `ablation_study.py` / `cross_corpus.py` runs also write `best_model.pth` per variant.
Those files are large and are **not** required to regenerate paper figures (figures use `history.csv`, `ablation_summary.csv`, confusion matrices).

**Headline test accuracy / macro F1:** bar labels on `ablation_comparison.png` → `ablation_summary.csv` (run `python3 sync_csv_from_ablation_pngs.py` after any relabel). Confusion-matrix top-1 alone is not used for ablation headlines.

If you trained on another machine, copy checkpoints here:

```text
results/Plant VIllage Result/<variant>/best_model.pth
results/Plant Doc Result/<variant>/best_model.pth
```

See `RUN_MANIFEST.json` for commands and completion timestamps.
