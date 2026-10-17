# ConvGeM-NeXt 🌿

PyTorch implementation of **ConvGeM-NeXt**, a deep learning framework for plant disease detection.

> Arshad Z., Javed A., Saudagar A.K.J. (2026). *ConvGeM-next: a deep learning framework for plant disease detection.* Frontiers in Plant Science, 17, 1763739. https://doi.org/10.3389/fpls.2026.1763739

---

## Architecture

Paper baseline (Experiment A):

```
Input (B×3×224×224)
        ↓
ConvNeXt-Base (ImageNet pre-trained, features only)
        ↓
Generalized Mean Pooling  (learnable exponent p, init=3.0)
        ↓
Linear(1024→1024) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(1024→num_classes)
        ↓
Logits (B×num_classes)
```

---

## Architectural Ablation Study

This repo compares **four architectures** that differ **only** in the neck
(feature aggregation). Dataset, 70/15/15 split, seed, ImageNet
normalisation, optimizer (AdamW 1e-3, weight decay 1e-4), cosine
annealing, 30 epochs, batch size 64, and CrossEntropyLoss are identical.

| Experiment | Variant | Question it answers |
|------------|---------|---------------------|
| **A** | `baseline` | What does the original ConvGeM-NeXt achieve? |
| **B** | `multiscale` | Does multi-scale information help? |
| **C** | `attention` | Does channel + spatial attention help? |
| **D** | `multiscale_attention` | Does combining both add further benefit? |

Do **not** treat any variant as better until you have actually trained
and evaluated it. Numbers in `results/` come from real runs only.

### Why these modifications

- **Multi-scale fusion (MSFF):** Stage 2 keeps finer spatial detail
  (lesion texture, leaf veins). Stage 3/4 add mid- and high-level
  disease context. Fusing them aims to preserve both grain and semantics.
- **Channel + spatial attention:** Channel attention up-weights
  disease-related feature maps; spatial attention focuses on infected
  regions and down-weights background. The block is a lightweight
  CBAM-style module (GAP-MLP + 7×7 spatial conv), not a transformer.

### ASCII diagrams

```
BASELINE
--------
ConvNeXt-Base
    ↓
Stage 4
    ↓
GeM
    ↓
Classifier


MULTI-SCALE
-----------
Stage 2 ─┐
Stage 3 ─┼→ Fusion → GeM → Classifier
Stage 4 ─┘


ATTENTION
---------
Stage 4
  ↓
Channel Attention
  ↓
Spatial Attention
  ↓
GeM
  ↓
Classifier


PROPOSED
--------
Stage 2 ─┐
Stage 3 ─┼→ Multi-Scale Fusion
Stage 4 ─┘
        ↓
Channel Attention
        ↓
Spatial Attention
        ↓
GeM
        ↓
Classifier
```

### Tensor sizes (224×224 input)

| Point | Shape |
|-------|--------|
| Stage 2 | `B×256×28×28` |
| Stage 3 | `B×512×14×14` |
| Stage 4 | `B×1024×7×7` |
| Fused map (B / D) | `B×256×28×28` |
| After GeM (A / C) | `B×1024` |
| After GeM (B / D) | `B×256` |
| Logits | `B×num_classes` |

Classifier: `Linear(C→1024) → BN → ReLU → Dropout(0.3) → Linear(1024→num_classes)`
where `C=1024` for A/C and `C=256` for B/D.

### How to run one experiment

```bash
python train.py --variant baseline --dataset plantvillage
python train.py --variant multiscale --dataset plantvillage
python train.py --variant attention --dataset plantvillage
python train.py --variant multiscale_attention --dataset plantvillage

python evaluate.py --variant baseline --dataset plantvillage
```

(`--dataset_name PlantVillage` still works.)

### How to run the full ablation

```bash
python ablation_study.py --dataset plantvillage
python ablation_study.py --dataset plantdoc
```

Smoke / pipeline check on the tiny synthetic set:

```bash
python sanity_check.py
python ablation_study.py --dataset demo --epochs 1 --batch_size 8 --no_pretrained
```

### Outputs

```
results/
  baseline/
    best_model.pth
    history.csv
    run_config.json
    metrics.json
    classification_report.csv
    confusion_matrix.png
  multiscale/
  attention/
  multiscale_attention/
  ablation_summary.csv
  ablation_comparison.png
```

`ablation_summary.csv` columns: Variant, Accuracy, Macro_F1, Weighted_F1,
Precision, Recall, Parameters, Best_Epoch, Training_Time.

Improvements change the **structure**, so parameter counts will differ
slightly across variants. That is expected.

### Teaching aid (tutors only)

If lab GPUs cannot finish a full ablation in time, tutors may use **synthetic**
example figures so students learn what typical curves and summary tables look
like. These are **not** substitutes for running experiments:

→ [`teaching_materials/README.md`](teaching_materials/README.md) — regenerate with
`python generate_teaching_figures.py`. Students still run `ablation_study.py` for
real metrics when hardware allows.

---

## Results (from paper)

| Dataset | Classes | Top-1 Acc | Macro F1 |
|---------|---------|-----------|----------|
| PlantVillage | 38 | **99.65%** | 0.9959 |
| PlantDoc | 27 | **94.69%** | 0.9465 |

---

## Project Structure

```
ConvGeM-NeXt/
├── config.py              # Hyperparameters, MODEL_VARIANT, DATASET aliases
├── model.py               # GeM, MSFF, attention, ConvGeMNeXt(variant=...)
├── dataset.py             # Loading + reusable 70/15/15 stratified splits
├── train.py               # Training loop (AMP, AdamW, CosineAnnealingLR)
├── evaluate.py            # Metrics + confusion matrix + metrics.json
├── ablation_study.py      # Train/eval all four variants
├── sanity_check.py        # Dummy forward/backward for all variants
├── create_demo_dataset.py
├── create_sample_test_dataset.py
├── create_realistic_dataset.py
├── create_balanced_dataset.py
├── requirements.txt
├── data/                  # PlantVillage/, PlantDoc/, SampleTest_*, RealisticPV/, Demo/
├── results/<variant>/     # Checkpoints, logs, metrics per experiment
├── checkpoints/           # Legacy demo path
└── logs/                  # Legacy demo path
```

---

## Quick Demo (no dataset download)

Use this to present the project to your teacher **right now** — no Kaggle download needed.

```bash
cd ConvGeM-NeXt

# Full mini demo: creates 100 synthetic images, trains 2 epochs, evaluates (~5–15 min on CPU)
python3 demo.py

# Fastest: only show architecture + data pipeline (~30 seconds)
python3 demo.py --no-train
```

The demo uses 5 synthetic plant-disease classes under `data/Demo/`. Full paper results use PlantVillage (38 classes) and PlantDoc (27 classes) once you download those datasets.

---

## SampleTest (synthetic, paper class names)

Use this when you want the **same 38 / 27 class folders** as PlantVillage / PlantDoc without downloading the real images. Images are generated locally; they are **not** a substitute for paper accuracy.

```bash
cd ConvGeM-NeXt

# Write data/SampleTest_PlantVillage and data/SampleTest_PlantDoc
python create_sample_test_dataset.py
python create_sample_test_dataset.py --force   # regenerate

# Train (pipeline check only — do not compare to paper numbers)
python train.py --dataset sampletest_plantvillage --epochs 2 --batch_size 8 --no_pretrained
python train.py --dataset sampletest_plantdoc --epochs 2 --batch_size 8 --no_pretrained
python ablation_study.py --dataset sampletest_plantvillage --epochs 1 --batch_size 8 --no_pretrained
```

Bare `sampletest` / `sample_test` resolve to `SampleTest_PlantVillage` (38 classes). Prefer the explicit aliases above. Canonical folder names: `SampleTest_PlantVillage`, `SampleTest_PlantDoc`.

---

## BalancedPV (medium synthetic — recommended for viva demo)

| Dataset | Difficulty | Typical CNN ablation | kNN proxy (no CNN) |
|---------|------------|----------------------|--------------------|
| **SampleTest_PlantVillage** | Easy (strong per-class hues) | Often ~100% quickly | ~85–90% |
| **BalancedPV** | Medium (crop hue bands + visible symptoms) | **~90–95% target** after 12 epochs | ~35–45% (kNN proxy) |
| **RealisticPV** | Hard (shared leaf, overlapping lesions) | Often near chance without long training | ~2–5% |

`data/BalancedPV/` uses the same **38 official PlantVillage folder names**. Images are synthetic leaf-on-soil JPEGs with learnable but non-cartoon cues — suitable for a fair four-variant ablation without SampleTest saturation or RealisticPV collapse. **~90–95% after training is a realistic demo goal, not a guarantee**; run training yourself and report whatever your run produces.

```bash
cd ConvGeM-NeXt

# 38 classes × 32 images (1216 total). Skip if data/BalancedPV already exists.
python create_balanced_dataset.py
python create_balanced_dataset.py --force   # regenerate

# Four-variant ablation (~1–2 h on RTX 4050)
python ablation_study.py --dataset balanced --epochs 12 --batch_size 32
```

Aliases `balanced`, `balancedpv`, and `balanced_pv` resolve to `BalancedPV` (38 classes).

---

## RealisticPV (harder synthetic, same 38 class names)

`data/RealisticPV/` is a larger synthetic set for a fair ablation when SampleTest saturates near 100%. Every class shares one leaf (same ellipse, veins, and soil). Disease cues are subtle and overlap related classes. Images are **not** real PlantVillage photos. Whatever accuracy training reports is the result of that run — not the paper's 99.65%.

```bash
cd ConvGeM-NeXt

# 38 classes × 40 images. Skip if data/RealisticPV already exists.
python create_realistic_dataset.py
python create_realistic_dataset.py --force   # regenerate

# Four-variant ablation (baseline, multiscale, attention, multiscale_attention).
# About 1–2 hours on an RTX 4050 at 12 epochs, batch 32.
python ablation_study.py --dataset realistic --epochs 12 --batch_size 32
```

Aliases `realistic`, `realisticpv`, and `realistic_pv` all resolve to `RealisticPV` (38 classes). Summary files: `results/ablation_summary.csv` and `results/ablation_comparison.png`.

---

## Improvements (for teacher / viva)

These go **beyond the baseline paper implementation**:

| Improvement | What it does | Why it matters |
|-------------|--------------|----------------|
| **Stratified K-Fold Cross-Validation** | Trains on K different splits, reports mean ± std accuracy/F1 | Single split can be lucky/unlucky; CV gives **reliable detection metrics** |
| **Cross-Class Misdetection Analysis** | Lists which diseases get confused (e.g. Tomato blight → Tomato healthy) | Shows **where detection fails across classes**, not just overall accuracy |
| **Mean Class Accuracy (MCA)** | Equal weight per disease class | Fair evaluation when some diseases have fewer images |
| **Demo pipeline** | Synthetic data + offline training | Runnable presentation without multi-GB downloads |

```bash
# Run cross-validation + cross-class analysis on demo data (~2–5 min CPU)
python cross_validation.py --dataset_name Demo --folds 3 --epochs 1 --no_pretrained
```

**Planned with full datasets:** cross-dataset generalization (train on PlantVillage → test on PlantDoc) to measure real-world detection transfer.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets and place them as:
#    data/PlantVillage/<class_name>/*.jpg
#    data/PlantDoc/<class_name>/*.jpg
#
#    Or generate a synthetic SampleTest with the same class folders:
#    python create_sample_test_dataset.py
#    → data/SampleTest_PlantVillage/  (38 classes)
#    → data/SampleTest_PlantDoc/      (27 classes)
#    Harder synthetic set (shared leaf, overlapping lesions):
#    python create_realistic_dataset.py
#    → data/RealisticPV/              (38 classes × 40 images)
```

**PlantVillage** — available on Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease  
**PlantDoc** — available on GitHub: https://github.com/pratikkayal/PlantDoc-Dataset

---

## Training

```bash
# Train on PlantVillage (38 classes), paper baseline
python train.py --dataset_name PlantVillage --epochs 30 --batch_size 64 --variant baseline

# Train on PlantDoc (27 classes)
python train.py --dataset_name PlantDoc --epochs 30 --batch_size 32 --variant baseline

# SampleTest (synthetic, official class names — pipeline only)
python train.py --dataset sampletest_plantvillage --epochs 2 --batch_size 8 --no_pretrained

# Override learning rate
python train.py --dataset_name PlantVillage --lr 5e-4
```

Training outputs (default):
- `results/<variant>/best_model.pth` — best model by validation loss
- `results/<variant>/history.csv` — train/val loss and accuracy
- `results/<variant>/run_config.json` — hyperparameters + parameter count

---

## Evaluation

```bash
# Evaluate PlantVillage checkpoint
python evaluate.py --dataset_name PlantVillage

# Evaluate PlantDoc checkpoint
python evaluate.py --dataset_name PlantDoc --checkpoint checkpoints/best_model.pth
```

Evaluation outputs:
- Console: Top-1 Acc, MCA, Macro P/R/F1, per-class report
- `logs/classification_report_<dataset>.txt`
- `logs/confusion_matrix_<dataset>.png`

---

## Key Design Choices

| Choice | Rationale |
|--------|-----------|
| No data augmentation | Paper isolates architectural gains from augmentation gains |
| GeM pooling (p=3.0, learnable) | Richer spatial summarisation vs. avg/max pool |
| AdamW + CosineAnnealingLR | Stable convergence with good weight regularisation |
| AMP (float16) | ~2× speed on modern GPUs with no accuracy loss |
| Stratified 70/15/15 split | Ensures rare disease classes appear in all splits |
