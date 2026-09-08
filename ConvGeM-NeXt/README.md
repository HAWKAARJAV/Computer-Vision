# ConvGeM-NeXt 🌿

PyTorch implementation of **ConvGeM-NeXt**, a deep learning framework for plant disease detection.

> Arshad Z., Javed A., Saudagar A.K.J. (2026). *ConvGeM-next: a deep learning framework for plant disease detection.* Frontiers in Plant Science, 17, 1763739. https://doi.org/10.3389/fpls.2026.1763739

---

## Architecture

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

## Results (from paper)

| Dataset | Classes | Top-1 Acc | Macro F1 |
|---------|---------|-----------|----------|
| PlantVillage | 38 | **99.65%** | 0.9959 |
| PlantDoc | 27 | **94.69%** | 0.9465 |

---

## Project Structure

```
ConvGeM-NeXt/
├── config.py          # All hyperparameters in one place
├── model.py           # GeMPooling + ConvGeMNeXt architecture
├── dataset.py         # Data loading + 70/15/15 stratified splits
├── train.py           # Training loop (AMP, AdamW, CosineAnnealingLR)
├── evaluate.py        # Metrics + confusion matrix
├── requirements.txt
├── data/              # → PlantVillage/ and PlantDoc/ go here
├── checkpoints/       # best_model.pth saved here
└── logs/              # CSV logs, loss curves, confusion matrices
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
```

**PlantVillage** — available on Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease  
**PlantDoc** — available on GitHub: https://github.com/pratikkayal/PlantDoc-Dataset

---

## Training

```bash
# Train on PlantVillage (38 classes)
python train.py --dataset_name PlantVillage --epochs 30 --batch_size 64

# Train on PlantDoc (27 classes)
python train.py --dataset_name PlantDoc --epochs 30 --batch_size 32

# Override learning rate
python train.py --dataset_name PlantVillage --lr 5e-4
```

Training outputs:
- `checkpoints/best_model.pth` — best model by validation loss
- `logs/training_log_<dataset>.csv` — epoch-level metrics
- `logs/loss_curve_<dataset>.png` — train vs val loss plot

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
