# Illustrative ablation figures for classroom demo

**These files are not from a live training run on student lab machines.** They are synthetic, pedagogical examples so learners can see what a typical four-variant ablation *looks like* when GPUs are too slow or unavailable for a full study.

Students must run `ablation_study.py` on their own data split when hardware allows and report **their** measured numbers for graded work. Do not submit these CSVs or plots as if they were your experiment results.

## Random seed

All curves and minor metric jitter are reproducible with:

```text
ILLUSTRATIVE_SEED = 20260417
```

(set in `generate_teaching_figures.py`)

Regenerate everything:

```bash
python generate_teaching_figures.py
```

## Contents

| File / folder | Description |
|---------------|-------------|
| `ablation_summary_ILLUSTRATIVE.csv` | Same columns as `results/ablation_summary.csv` (Variant, Accuracy, Macro_F1, …) with a teaching narrative: baseline lowest → multiscale → attention → multiscale_attention best (~88–96%, not perfect 100%). |
| `ablation_comparison_ILLUSTRATIVE.png` | Grouped bar chart: Accuracy vs Macro F1 across variants. |
| `baseline/` | `history_ILLUSTRATIVE.csv`, `loss_curve_ILLUSTRATIVE.png`, `accuracy_curve_ILLUSTRATIVE.png` (20 epochs, realistic val noise). |
| `multiscale/` | Same as above for the multiscale variant. |
| `attention/` | Same as above for the attention variant. |
| `multiscale_attention/` | Same as above, plus `confusion_matrix_ILLUSTRATIVE.png` (generic class labels, not a real dataset export). |

Every plot title includes **ILLUSTRATIVE** and states that values were not measured from the viewer’s run.

## For tutors

1. Show `ablation_comparison_ILLUSTRATIVE.png` first — discuss *relative* ordering and small gaps, not memorising exact percentages.
2. Open one variant’s accuracy curve and point out smooth train vs bumpier val; compare baseline vs `multiscale_attention`.
3. Open the summary CSV side-by-side with a student’s real `results/ablation_summary.csv` when they finish training — contrast “example shape” vs “your numbers”.
4. Emphasise that parameter counts in the illustrative CSV match real model sizes, but accuracies are **fabricated for teaching**.

Nothing in this folder claims results from PlantVillage, PlantDoc, BalancedPV, or RealisticPV training.
