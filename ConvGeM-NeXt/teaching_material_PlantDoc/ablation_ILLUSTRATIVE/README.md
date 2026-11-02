# PlantDoc ablation — ILLUSTRATIVE demo

**ILLUSTRATIVE DEMO — not from a real PlantDoc training run**

Same folder layout as `results/` from `ablation_study.py`, for classroom walkthroughs when a full PlantDoc ablation has not finished yet. Baseline test accuracy matches the paper (**94.69 %**); each next variant adds a small gain from our lab ablation gaps (scaled): baseline: 94.69 % → multiscale: 95.65 % → attention: 96.14 % → multiscale_attention: 96.80 %.

Regenerate:

```bash
python generate_plantdoc_ablation_illustrative.py
```

For graded work, run `python ablation_study.py --dataset plantdoc` and use those files instead.
