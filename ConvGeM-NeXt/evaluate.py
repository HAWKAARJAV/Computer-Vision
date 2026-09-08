"""
evaluate.py — Evaluation Script for ConvGeM-NeXt
==================================================
Loads the saved best checkpoint and computes:
  * Top-1 Accuracy
  * Mean Class Accuracy  (MCA — arithmetic mean of per-class accuracies)
  * Macro-averaged Precision, Recall, F1-score
  * Full per-class classification report  (via scikit-learn)
  * Confusion matrix  (plotted as a heatmap and saved as PNG)

Can be run on the PlantVillage test set or the PlantDoc test set
by passing --dataset_name PlantDoc (and a matching checkpoint).

Usage
-----
  # Evaluate the saved checkpoint on PlantVillage:
  python evaluate.py --dataset_name PlantVillage

  # Evaluate on PlantDoc:
  python evaluate.py --dataset_name PlantDoc --checkpoint checkpoints/best_model.pth
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import config as cfg
from model import ConvGeMNeXt
from dataset import get_dataloaders


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------

def mean_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Compute Mean Class Accuracy (MCA).

    MCA is the arithmetic mean of per-class recall values.  Unlike global
    accuracy it treats every class equally, regardless of sample count,
    making it particularly informative on imbalanced disease datasets.

    MCA = (1/C) * Σ_i  (TP_i / N_i)

    where TP_i is the number of correctly classified samples for class i
    and N_i is the total number of samples in class i.

    Args:
        y_true:      Ground-truth integer labels.
        y_pred:      Predicted integer labels.
        num_classes: Total number of classes C.

    Returns:
        Mean class accuracy as a float in [0, 1].
    """
    class_accs = []
    for cls in range(num_classes):
        mask = y_true == cls
        if mask.sum() == 0:
            continue                      # skip classes absent from this split
        cls_acc = (y_pred[mask] == cls).mean()
        class_accs.append(cls_acc)
    return float(np.mean(class_accs))


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
    dataset_name: str,
) -> None:
    """Plot a normalised confusion matrix heatmap and save as PNG.

    Normalises each row (true class) by the total number of samples so that
    cell values represent per-class recall (same scale regardless of class size).

    Args:
        cm:           Raw integer confusion matrix (shape [C, C]).
        class_names:  List of class label strings.
        save_path:    File path to save the PNG.
        dataset_name: Used in the plot title.
    """
    # Row-normalise: each row sums to 1 (per-class recall scale)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    n = len(class_names)
    # Dynamically size figure: bigger for more classes
    fig_size = max(10, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm_norm,
        annot=(n <= 40),          # show values only if ≤40 classes (readability)
        fmt=".2f" if n <= 40 else "",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"shrink": 0.7},
        linewidths=0.3 if n <= 40 else 0,
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(f"Confusion Matrix — {dataset_name} (row-normalised)", fontsize=13)
    plt.xticks(rotation=90, fontsize=max(4, 10 - n // 10))
    plt.yticks(rotation=0,  fontsize=max(4, 10 - n // 10))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Inference pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """Run inference on all batches and collect predictions + ground-truth.

    Args:
        model:  ConvGeMNeXt in eval mode.
        loader: DataLoader for the evaluation split.
        device: Compute device.

    Returns:
        Tuple ``(y_true, y_pred)`` as numpy arrays of shape ``(N,)``.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Load checkpoint and compute all evaluation metrics on the test split."""

    # ---- Device ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print(f"  ConvGeM-NeXt Evaluation")
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Dataset    : {args.dataset_name}")
    print(f"  Checkpoint : {args.checkpoint}")
    print("="*60 + "\n")

    # ---- Load checkpoint ---------------------------------------------------
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: '{args.checkpoint}'\n"
            "Train the model first with: python train.py"
        )

    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt.get("num_classes", cfg.get_num_classes(args.dataset_name))
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} | "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f} | "
          f"val_acc={ckpt.get('val_accuracy', float('nan'))*100:.2f}%\n")

    # ---- Model -------------------------------------------------------------
    model = ConvGeMNeXt(
        num_classes=num_classes,
        p_init=cfg.GEM_P_INIT,
        dropout=cfg.DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Data loaders ------------------------------------------------------
    num_workers = cfg.NUM_WORKERS if device.type == "cuda" else 0
    pin_memory  = cfg.PIN_MEMORY  if device.type == "cuda" else False

    _, _, test_loader, full_dataset = get_dataloaders(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        img_size=cfg.IMG_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
        random_seed=cfg.RANDOM_SEED,
    )
    class_names = full_dataset.classes

    # ---- Collect predictions -----------------------------------------------
    print("Running inference on test set …")
    y_true, y_pred = collect_predictions(model, test_loader, device)

    # ---- Compute metrics ---------------------------------------------------
    top1_acc  = accuracy_score(y_true, y_pred)
    mca       = mean_class_accuracy(y_true, y_pred, num_classes)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ---- Print summary metrics --------------------------------------------
    print("\n" + "="*60)
    print(f"  Test-set Results — {args.dataset_name}")
    print("="*60)
    print(f"  Top-1 Accuracy          : {top1_acc*100:.2f}%")
    print(f"  Mean Class Accuracy     : {mca*100:.2f}%")
    print(f"  Macro Precision         : {macro_precision:.4f}")
    print(f"  Macro Recall            : {macro_recall:.4f}")
    print(f"  Macro F1-Score          : {macro_f1:.4f}")
    print("="*60 + "\n")

    # ---- Full per-class classification report ------------------------------
    print("Per-class Classification Report:")
    print("-" * 60)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    # Optionally save report to a text file
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(args.logs_dir, f"classification_report_{args.dataset_name}.txt")
    with open(report_path, "w") as f:
        f.write(f"ConvGeM-NeXt — Test Results ({args.dataset_name})\n")
        f.write("="*60 + "\n")
        f.write(f"Top-1 Accuracy      : {top1_acc*100:.2f}%\n")
        f.write(f"Mean Class Accuracy : {mca*100:.2f}%\n")
        f.write(f"Macro Precision     : {macro_precision:.4f}\n")
        f.write(f"Macro Recall        : {macro_recall:.4f}\n")
        f.write(f"Macro F1-Score      : {macro_f1:.4f}\n")
        f.write("\nPer-class Report:\n")
        f.write(report)
    print(f"Report saved → {report_path}")

    # ---- Confusion matrix --------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(args.logs_dir, f"confusion_matrix_{args.dataset_name}.png")
    plot_confusion_matrix(cm, class_names, cm_path, args.dataset_name)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ConvGeM-NeXt checkpoint on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=cfg.DATASET_NAME,
        choices=["PlantVillage", "PlantDoc"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=cfg.DATA_DIR,
        help="Root directory containing dataset sub-folders.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"),
        help="Path to the model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=cfg.BATCH_SIZE,
        help="Batch size for inference (can be larger than training batch size).",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=cfg.LOGS_DIR,
        help="Directory to save evaluation outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
