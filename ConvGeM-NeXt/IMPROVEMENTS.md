# Proposed improvements over ConvGeM-NeXt (Arshad et al., 2026)

This document lists changes that target weaknesses the paper itself reports. Each one is implemented as an opt-in flag. With no flags set, the code reproduces the paper's recipe. `improvement_study.py` measures each change cumulatively, using the same split, seed and test set.

**No accuracy is claimed here.** Fill the results table from `results/improvements/<Dataset>/<variant>/improvement_summary.csv` after running the study.

---

## 1. Weaknesses found in the paper

| # | Observation in the paper | Where |
|---|--------------------------|-------|
| W1 | No data augmentation at all, even though PlantDoc's difficulty comes from lighting, background, occlusion and pose. | §3.1 |
| W2 | Lab → field transfer is poor: 41 % PV→PD and 53.23 % PD→PV. PlantVillage is never used to *initialise* PlantDoc training, only for zero-shot testing. | §4.9, Table 14 |
| W3 | Class imbalance on PlantDoc (42–238 images per class). MCA is 87.04 % while accuracy is 94.69 %. Three classes have F1 = 0. | Table 5, §4.4.1, Table 8 |
| W4 | Healthy-leaf classes collapse under domain shift, which suggests the model relies on background and illumination cues. | §4.9.1, Table 17 |
| W5 | The whole ImageNet-pre-trained backbone is fine-tuned at LR 1e-3 with no warm-up. PlantDoc runs vary by ±2.58 % across seeds. | Table 2, Table 7 |
| W6 | Lesions are small, but the input is fixed at 224 px. | §3.1 |

### Internal inconsistencies worth noting (for your related-work or discussion section)

- **Table 8 vs. headline macro F1.** The per-class PlantDoc F1 values in Table 8 average to about **0.827**: 20 classes at 1.00, plus 0.667, 0.50, 0.667, 0.50, 0, 0, 0. The paper reports **0.9465**.
- **PV→PD accuracy.** The text says 42.0 %. Table 14 says 41.00 %.
- **PlantDoc size.** The text says 2,598 images (§4.1.2). Table 5 totals 2,922.
- **Model selection.** §3.3 says results come from the *last* epoch. Table 2 says *lowest validation loss*.
- **Test-set size.** With 2,598 images split 70/15/15, many PlantDoc classes have only a handful of test images. Per-class 100 % / 0 % values (Table 8) are therefore high-variance. Use the official PlantDoc test split and report mean ± std over seeds.

---

## 2. Proposed improvements

| Step | Change | Targets | Flag |
|------|--------|---------|------|
| S1 | **Field-robust augmentation**: RandomResizedCrop (0.5–1.0), H/V flips, ±20° rotation, ColorJitter, RandomErasing. Applied to the train split only. | W1, W4 | `--augment` |
| S2 | **Label smoothing 0.1**: reduces over-confidence on noisy field labels. | W3, W5 | `--label_smoothing 0.1` |
| S3 | **Class-balanced sampling**: inverse-frequency sampling of the train split. | W3 (MCA, zero-F1 classes) | `--balanced_sampler` |
| S4 | **Discriminative LR + warm-up**: backbone LR ×0.1, 2 warm-up epochs before cosine decay. | W5 | `--backbone_lr_mult 0.1 --warmup_epochs 2` |
| S5 | **PlantVillage → PlantDoc initialisation**: start PlantDoc training from the PlantVillage model instead of ImageNet. | W2 | `--init_checkpoint results/baseline/best_model.pth` |
| S6 | **Test-time augmentation**: average softmax over original + H-flip + V-flip. | W4 | `evaluate.py --tta` |
| opt. | **Higher resolution** (288 px) for small lesions. | W6 | `--img_size 288` |
| opt. | **Architecture**: multi-scale fusion and/or CBAM attention (already in this repo). | W4, W6 | `--variant multiscale_attention` |

Cross-corpus protocols (`cross_corpus.py`) extend Table 14 with *fine-tune* and *scratch* settings. These show how much of the lab→field gap S5 closes.

---

## 3. How to run (real measurements)

```bash
# 1. PlantVillage model (needed for S5 and cross-corpus)
python train.py --dataset plantvillage --variant baseline

# 2. Cumulative improvement study on PlantDoc (S0 = paper reproduction)
python improvement_study.py --dataset plantdoc --variant baseline \
    --pv_checkpoint results/baseline/best_model.pth

# 3. Repeat with 3 seeds (change RANDOM_SEED in config.py) and report mean ± std, as in the paper's Table 7

# 4. Cross-corpus table (extends the paper's Table 14)
python cross_corpus.py --source plantvillage --target plantdoc --variant baseline \
    --source_checkpoint results/baseline/best_model.pth
python cross_corpus.py --source plantdoc --target plantvillage --variant baseline
```

Outputs to put in your paper:
- `results/improvements/PlantDoc/<variant>/improvement_summary.csv` and `improvement_comparison.png`. These are in the same form as the paper's Table 10 ablation.
- Per-step `confusion_matrix.png` and `classification_report.csv`, for comparison with Figure 7 and Table 8.
- `results/cross_corpus/.../cross_corpus_summary.csv`, for comparison with Table 14.

## 4. Results table (fill from your runs)

| Step | Description | Accuracy (%) | Macro F1 | MCA (%) | Δ vs S0 |
|------|-------------|--------------|----------|---------|---------|
| Paper | ConvGeM-NeXt (reported) | 94.69 | 0.9465 | 87.04 | — |
| S0 | Paper recipe (our reproduction) | | | | |
| S1 | + augmentation | | | | |
| S2 | + label smoothing | | | | |
| S3 | + balanced sampling | | | | |
| S4 | + discriminative LR / warm-up | | | | |
| S5 | + PlantVillage initialisation | | | | |
| S6 | + TTA | | | | |

Compare S1–S6 with **your own S0**, not only with the paper's 94.69 %. Splits and hardware differ (TPU vs. GPU), so the fair claim is the gain over your reproduction.
