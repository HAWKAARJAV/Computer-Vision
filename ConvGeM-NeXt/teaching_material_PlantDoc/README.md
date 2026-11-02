# PlantDoc teaching results (from the published paper)

Every number comes from **Arshad et al., "ConvGeM-next: a deep learning framework for plant disease detection", Front. Plant Sci. 17:1763739 (2026)**. No training was run and no values were invented. Every CSV row and every plot names its source table or figure.

The layout mirrors `results/` as written by `ablation_study.py`, so students can put these files next to their own runs. Regenerate with `python generate_plantdoc_teaching_figures.py`.

For the **four-variant architectural ablation** (baseline → multiscale → attention → multiscale_attention) with loss/accuracy curves like a real run, see **`ablation_ILLUSTRATIVE/`**. Those files are **demo-only** (banner on every plot, `Note=ILLUSTRATIVE_DEMO` in the CSV). Regenerate with `python generate_plantdoc_ablation_illustrative.py`. Do not submit them as measured experiment results.

```
teaching_material_PlantDoc/
  ablation_summary.csv          same columns as results/ablation_summary.csv, plus Source
  ablation_comparison.png       Accuracy vs Macro F1: ConvNeXt-Base vs ConvGeM-NeXt
  model_comparison.{csv,png}    PlantDoc accuracy of all models in Table 11
  convnext_base/metrics.json    plain ConvNeXt-Base (§4.3: 91.84 %, F1 0.9231)
  baseline/                     ConvGeM-NeXt (= this repo's "baseline" variant)
    metrics.json                94.69 % acc, 0.9477 prec, 0.9465 F1, 87.04 % MCA, loss 0.2552, 3-seed 92.19 ± 2.58 %
    classification_report.csv   per-class P/R/F1 (Table 8), reported macro avg, and macro avg recomputed from Table 8
    classwise_f1.png
    confusion_matrix.{csv,png}  row-normalised %, reconstructed from Fig. 7 / Table 8
    history.csv, loss_curve.png loss only, an approximate by-eye trace of Fig. 5
  ablation_ILLUSTRATIVE/          four-variant demo (baseline 94.69 % paper, then +gains), banner on every PNG
    ablation_summary.csv, ablation_comparison.png
    baseline|multiscale|attention|multiscale_attention/
      history.csv, loss_curve.png, accuracy_curve.png, metrics.json
    multiscale_attention/ also confusion_matrix.{csv,png}
```

## Differences from a real `results/` run

- **Blank cells** mean the paper doesn't report that value (Weighted_F1, Recall, Best_Epoch, Training_Time, Inference_ms, per-class support, accuracy curves). They are left blank on purpose.
- **Paper folder has no measured multiscale / attention / multiscale_attention runs.** Use `ablation_ILLUSTRATIVE/` for a labelled classroom demo, or run `python ablation_study.py --dataset plantdoc` for real metrics.
- **The confusion matrix is in percentages, not counts.** Fig. 7 is row-normalised.
- **Parameters (88,655,015)** are counted from `model.py`. The paper says "about 89 million".

## Teaching points

1. **94.69 % is a strong field result.** Every other model in Table 11 scores 47–78 % on PlantDoc.
2. **Look at the variance, not just the best run.** The three-seed mean is 92.19 ± 2.58 % (Table 7).
3. **Accuracy vs. class balance.** MCA is 87.04 %. Potato late blight, tomato mosaic virus and tomato yellow virus have F1 = 0. The confusion matrix shows where those images are predicted instead.
4. **Read tables critically.** Averaging Table 8 gives macro recall **0.8704**, which matches the reported MCA. It gives macro F1 **0.8272**, while the paper reports 0.9465. Both rows are in `baseline/classification_report.csv`. Ask students which number to trust, and why.
5. **Lab ≠ field.** Trained only on PlantVillage, the model reaches 41 % on PlantDoc (Table 14).
