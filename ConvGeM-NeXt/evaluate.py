"""
evaluate.py — Evaluation Script for ConvGeM-NeXt
==================================================
Loads the saved best checkpoint and computes:
  * Top-1 Accuracy
  * Precision / Recall / F1  (macro and weighted)
  * Mean Class Accuracy  (MCA)
  * Per-class classification report
  * Confusion matrix heatmap

Outputs (per variant):
  results/<variant>/metrics.json
  results/<variant>/classification_report.csv
  results/<variant>/confusion_matrix.png
"""

import csv
import json
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
from model import ConvGeMNeXt, count_parameters
from dataset import get_dataloaders


def mean_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Compute Mean Class Accuracy (MCA)."""
    class_accs = []
    for cls in range(num_classes):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        cls_acc = (y_pred[mask] == cls).mean()
        class_accs.append(cls_acc)
    return float(np.mean(class_accs)) if class_accs else 0.0


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
    dataset_name: str,
) -> None:
    """Plot a normalised confusion matrix heatmap and save as PNG."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    n = len(class_names)
    fig_size = max(10, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm_norm,
        annot=(n <= 40),
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


def save_classification_report_csv(report_dict: dict, class_names: list, save_path: str) -> None:
    fieldnames = ["class", "precision", "recall", "f1-score", "support"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in class_names:
            row = report_dict.get(name, {})
            writer.writerow({
                "class": name,
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "f1-score": row.get("f1-score", ""),
                "support": row.get("support", ""),
            })
        for extra in ("macro avg", "weighted avg"):
            row = report_dict.get(extra, {})
            writer.writerow({
                "class": extra,
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "f1-score": row.get("f1-score", ""),
                "support": row.get("support", ""),
            })
    print(f"[evaluate] Classification report CSV saved → {save_path}")


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """Run inference on all batches and collect predictions + ground-truth."""
    model.eval()
    all_preds = []
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mca": mean_class_accuracy(y_true, y_pred, num_classes),
    }


def run_evaluation(args: argparse.Namespace) -> dict:
    """Load checkpoint and compute all evaluation metrics on the test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg.resolve_dataset_name(args.dataset_name)

    print("\n" + "=" * 60)
    print("  ConvGeM-NeXt Evaluation")
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Dataset    : {dataset_name}")
    print(f"  Checkpoint : {args.checkpoint}")
    print("=" * 60 + "\n")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: '{args.checkpoint}'\n"
            "Train the model first with: python train.py --variant <name>"
        )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", cfg.get_num_classes(dataset_name))
    variant = args.variant or ckpt.get("variant", cfg.MODEL_VARIANT)
    variant = cfg.resolve_variant(variant)
    no_pretrained = args.no_pretrained or ckpt.get("no_pretrained", False)
    print(
        f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} | "
        f"variant={variant} | "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f} | "
        f"val_acc={ckpt.get('val_accuracy', float('nan'))*100:.2f}%\n"
    )

    output_dir = args.output_dir or os.path.join(args.results_dir, variant)
    logs_dir = args.logs_dir or output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    model = ConvGeMNeXt(
        num_classes=num_classes,
        variant=variant,
        p_init=cfg.GEM_P_INIT,
        dropout=cfg.DROPOUT,
        pretrained=not no_pretrained,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    total_params, _ = count_parameters(model)

    num_workers = cfg.NUM_WORKERS if device.type == "cuda" else 0
    pin_memory = cfg.PIN_MEMORY if device.type == "cuda" else False

    _, _, test_loader, full_dataset = get_dataloaders(
        data_dir=args.data_dir,
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        img_size=cfg.IMG_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
        random_seed=cfg.RANDOM_SEED,
    )
    class_names = full_dataset.classes

    print("Running inference on test set …")
    y_true, y_pred = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, num_classes)
    metrics.update({
        "variant": variant,
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "parameters": int(ckpt.get("parameters", total_params)),
        "best_epoch": int(ckpt.get("epoch", 0)),
        "val_loss": float(ckpt.get("val_loss", float("nan"))),
        "val_accuracy": float(ckpt.get("val_accuracy", float("nan"))),
    })

    print("\n" + "=" * 60)
    print(f"  Test-set Results — {dataset_name} [{variant}]")
    print("=" * 60)
    print(f"  Top-1 Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"  Mean Class Accuracy     : {metrics['mca']*100:.2f}%")
    print(f"  Precision (macro)       : {metrics['precision']:.4f}")
    print(f"  Recall (macro)          : {metrics['recall']:.4f}")
    print(f"  F1 (macro)              : {metrics['macro_f1']:.4f}")
    print(f"  F1 (weighted)           : {metrics['weighted_f1']:.4f}")
    print("=" * 60 + "\n")

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    print("Per-class Classification Report:")
    print("-" * 60)
    print(report)

    report_txt = os.path.join(logs_dir, f"classification_report_{dataset_name}.txt")
    with open(report_txt, "w") as f:
        f.write(f"ConvGeM-NeXt — Test Results ({dataset_name}, {variant})\n")
        f.write("=" * 60 + "\n")
        for key in ("accuracy", "precision", "recall", "macro_f1", "weighted_f1", "mca"):
            f.write(f"{key}: {metrics[key]}\n")
        f.write("\nPer-class Report:\n")
        f.write(report)
    print(f"Report saved → {report_txt}")

    report_csv = os.path.join(output_dir, "classification_report.csv")
    save_classification_report_csv(report_dict, class_names, report_csv)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved → {metrics_path}")

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path, f"{dataset_name} [{variant}]")

    # Keep the older filename as well when writing into a logs/ folder
    legacy_cm = os.path.join(logs_dir, f"confusion_matrix_{dataset_name}.png")
    if os.path.abspath(legacy_cm) != os.path.abspath(cm_path):
        plot_confusion_matrix(cm, class_names, legacy_cm, f"{dataset_name} [{variant}]")

    metrics["metrics_path"] = metrics_path
    metrics["output_dir"] = output_dir
    return metrics


def main(args: argparse.Namespace) -> None:
    run_evaluation(args)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ConvGeM-NeXt checkpoint on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Alias for --dataset_name (plantvillage | plantdoc | demo | "
             "sampletest_plantvillage | sampletest_plantdoc | balanced | realistic).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Architecture variant. If omitted, read from the checkpoint.",
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
        default=None,
        help="Path to the model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=cfg.BATCH_SIZE,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=None,
        help="Optional extra directory for text reports.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=cfg.RESULTS_DIR,
        help="Root folder for per-variant outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Exact folder for metrics.json / report / confusion matrix.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Skip ImageNet backbone weights (must match how the model was trained).",
    )
    args = parser.parse_args()
    args.dataset_name = args.dataset_name or args.dataset or cfg.DATASET_NAME
    if args.checkpoint is None:
        variant = args.variant or cfg.MODEL_VARIANT
        try:
            variant = cfg.resolve_variant(variant)
        except Exception:
            variant = cfg.MODEL_VARIANT
        args.checkpoint = os.path.join(cfg.RESULTS_DIR, variant, "best_model.pth")
        # Fall back to the original default if the new path is missing
        if not os.path.isfile(args.checkpoint):
            legacy = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
            if os.path.isfile(legacy):
                args.checkpoint = legacy
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
