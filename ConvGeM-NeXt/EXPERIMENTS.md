# Experiments guide: from datasets to every figure

This guide covers everything needed to produce each graph and table in the paper from **real runs**. Every figure is rebuilt by `make_figures.py` from the files the runs write. Nothing is typed in by hand. Real runs write to `runs/`, which is kept separate from `results/`. The files in `results/` were not produced by `run_experiments.sh`.

```
prepare_datasets.py ─► data/ ─► run_experiments.sh e1…e5 ─► runs/ ─► make_figures.py ─► runs/figures/
```

---

## 0. Setup (once)

```bash
git clone https://github.com/HAWKAARJAV/ConvGeM-NeXt.git && cd ConvGeM-NeXt/ConvGeM-NeXt
pip install -r requirements.txt kaggle
# Kaggle API token: kaggle.com → Settings → API → "Create New Token" → save as ~/.kaggle/kaggle.json
```

To check that the whole pipeline works before using a GPU (about 1 hour on a laptop CPU, random-pixel data, numbers meaningless):

```bash
SMOKE=1 bash run_experiments.sh all     # writes data_smoke/ and results_smoke/
```

---

## 1. Datasets

| | PlantVillage | PlantDoc |
|---|---|---|
| Source | Kaggle `abdallahalidev/plantvillage-dataset` (the `color` folder) | GitHub `pratikkayal/PlantDoc-Dataset` |
| Images | 54,284 | 2,598 (paper §4.1.2) or 2,922 (paper Table 5); `--verify` reports the real count |
| Classes | 38 | 27 (the 2-image spider-mite class exists only in `train/` and is dropped) |
| Conditions | Lab: plain background, uniform light | Field: clutter, occlusion, variable light |
| Split | Stratified 70 / 15 / 15 → 37,998 / 8,143 / 8,143 | Official `test/` kept; 15 % of `train/` → validation |
| On disk | `data/PlantVillage/<class>/*.jpg` | `data/PlantDoc/{train,test}/<class>/*.jpg` |

```bash
bash run_experiments.sh data
# equivalent to:
python prepare_datasets.py --plantvillage --plantdoc   # download + arrange
python prepare_datasets.py --verify                    # counts, checks, split files
```

`--verify` writes `data/dataset_report.md`, which lists per-class counts, totals compared with the paper, the PlantDoc→PlantVillage class mapping and the split sizes. It also creates `data/PlantVillage_split_seed42.json` and `data/PlantDoc_split_seed42.json`, so every run uses the **same** images for train, val and test.

If you downloaded the data yourself, use `--pv_source <zip|folder>` / `--pd_source <zip|folder>`.

---

## 2. Experiments and the figure each one produces

| Step | Experiment | Train on → test on | Command | Main outputs | Figure |
|---|---|---|---|---|---|
| **E1** | PlantVillage ablation, 4 variants | PV train → PV test | `bash run_experiments.sh e1` | `runs/plantvillage/ablation_summary.csv`, `<variant>/{best_model.pth, history.csv, metrics.json, confusion_matrix.png, classification_report.csv}` | `fig_E1_plantvillage.png` |
| **E2** | PlantDoc ablation, 4 variants | PD train → PD test | `bash run_experiments.sh e2` | `runs/plantdoc/…` (same files) | `fig_E2_plantdoc.png` |
| **E3** | Cross-corpus PV → PD, zero-shot (paper §4.9) | E1 checkpoints → PD test | `bash run_experiments.sh e3` | `runs/cross_corpus/PlantVillage_to_PlantDoc/<variant>/direct/` | `fig_E3_E4_cross_corpus.png` (left) |
| **E4** | Cross-corpus PD → PV, zero-shot | E2 checkpoints → PV test | `bash run_experiments.sh e4` | `runs/cross_corpus/PlantDoc_to_PlantVillage/<variant>/direct/` | `fig_E3_E4_cross_corpus.png` (right) |
| **E5** | Training improvements S0–S6 on the proposed model | PD (S5 starts from the E1 checkpoint) | `bash run_experiments.sh e5` | `runs/improvements/PlantDoc/multiscale_attention/improvement_summary.csv` | `fig_E5_improvements_*.png` |
| — | All figures + one table of every measured number | — | `bash run_experiments.sh figures` | `runs/figures/*.png`, `runs/figures/all_results.csv` | `fig_summary.png` |

Order: `data → e1 → e2 → e3 → e4 → e5 → figures`, or `bash run_experiments.sh all`. E3 and E5 need E1, and E4 needs E2, because they reuse those checkpoints instead of retraining.

### What each variant is

| Variant | Neck | Parameters |
|---|---|---|
| `baseline` | ConvNeXt-Base stage 4 → GeM (the paper's model) | 88,655,015 |
| `multiscale` | stages 2–4 → multi-scale fusion → GeM | 90,097,831 |
| `attention` | stage 4 → CBAM channel + spatial attention → GeM | 88,787,274 |
| `multiscale_attention` (**proposed**) | stages 2–4 → fusion → CBAM → GeM | 90,106,394 |

### Training parameters (identical for every variant: paper Table 2)

224×224 input · ImageNet normalisation · no augmentation · AdamW, LR 1e-3, weight decay 1e-4 · cosine annealing, T<sub>max</sub> = 30 · 30 epochs · batch 64 (PV) / 32 (PD) · dropout 0.3 · GeM p = 3 (learnable) · cross-entropy · best checkpoint = lowest validation loss · seed 42 · AMP on GPU.

Override any of them with environment variables, for example `EPOCHS=15 BATCH_PV=32 bash run_experiments.sh e1`.

### E5 steps (cumulative, see `IMPROVEMENTS.md`)

S0 paper recipe → S1 + augmentation → S2 + label smoothing 0.1 → S3 + class-balanced sampling → S4 + backbone LR ×0.1 and 2 warm-up epochs → S5 + PlantVillage initialisation → S6 + test-time augmentation.

---

## 3. Time and hardware (rough guide)

| Step | Runs | Colab T4 (16 GB) | A100 |
|---|---|---|---|
| data | download ~1 GB + verify | 10–20 min | same |
| E1 PlantVillage | 4 × 30 epochs on 38k images | **~15–20 h** (use `EPOCHS=10` or an A100) | ~4–6 h |
| E2 PlantDoc | 4 × 30 epochs on ~1.8k images | ~1–1.5 h | ~20 min |
| E3 + E4 | 8 evaluations, no training | ~15 min | ~5 min |
| E5 | 6 × 30 epochs on PlantDoc | ~1.5–2 h | ~30 min |

These are estimates; the actual time per epoch is printed during training. Colab sessions time out, so run one step per session and copy `runs/` to Google Drive between steps (the notebook does this).

---

## 4. Reporting

- Use `runs/figures/all_results.csv` as the single source for every number in the paper. Each row names the file it came from.
- Run E2 with three seeds (change `RANDOM_SEED` in `config.py`) and report mean ± std, as the paper does in Table 7.
- Compare against your own `baseline` run on the same split, as well as against the paper's published values (dashed lines in the figures).
- `results/` (`Plant VIllage Result/`, `Plant Doc Result/`, `Cross_Corpus/`) and the demo papers hold **illustrative** values: see the `ILLUSTRATIVE_DEMO` note in their CSV/JSON files. Never copy numbers from them into the paper; use `runs/`.
