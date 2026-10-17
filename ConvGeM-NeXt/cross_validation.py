"""
cross_validation.py — Cross-validation & cross-class detection analysis
=========================================================================
Improvements over a single train/val/test split:

  1. Stratified K-Fold Cross-Validation
     Trains/evaluates on K different splits and reports mean ± std metrics.
     More reliable than one lucky/unlucky split.

  2. Cross-Class Misdetection Analysis
     Finds which disease classes get confused with each other (off-diagonal
     confusion matrix pairs) — helps improve detection on similar diseases.

Usage (demo dataset, fast):
    python cross_validation.py --dataset_name Demo --folds 3 --epochs 1

Full run (when real data is available):
    python cross_validation.py --dataset_name PlantVillage --folds 5 --epochs 30
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config as cfg
from dataset import load_imagefolder, make_loader
from evaluate import collect_predictions, mean_class_accuracy
from model import ConvGeMNeXt
from train import evaluate, train_one_epoch


def find_cross_class_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    top_k: int = 10,
) -> list:
    """Return the most frequent cross-class misdetections (true → predicted)."""
    pairs = {}
    for t, p in zip(y_true, y_pred):
        if t == p:
            continue
        key = (class_names[t], class_names[p])
        pairs[key] = pairs.get(key, 0) + 1

    ranked = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
    return [(true_cls, pred_cls, count) for (true_cls, pred_cls), count in ranked[:top_k]]


def train_fold(
    train_loader,
    val_loader,
    num_classes: int,
    epochs: int,
    lr: float,
    device: torch.device,
    no_pretrained: bool,
    variant: str = "baseline",
) -> tuple:
    """Train one CV fold and return (model, val_metrics)."""
    model = ConvGeMNeXt(
        num_classes=num_classes,
        variant=variant,
        p_init=cfg.GEM_P_INIT,
        dropout=cfg.DROPOUT,
        pretrained=not no_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_state = None
    best_val_loss = float("inf")
    best_metrics = {}

    for _ in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)  # loss, acc
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"val_loss": val_loss, "val_accuracy": val_acc}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 62)
    print("  Cross-Validation + Cross-Class Detection Analysis")
    print("=" * 62)
    print(f"  Dataset : {args.dataset_name}")
    print(f"  Folds   : {args.folds}")
    print(f"  Epochs  : {args.epochs} per fold")
    print(f"  Device  : {device}\n")

    dataset = load_imagefolder(args.data_dir, args.dataset_name, cfg.IMG_SIZE)
    labels = np.array(dataset.targets)
    num_classes = len(dataset.classes)
    n_samples = len(labels)

    if args.folds > n_samples // num_classes:
        raise ValueError(
            f"Need at least one sample per class per fold. "
            f"Try --folds {max(2, n_samples // num_classes // 2)} or fewer."
        )

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=cfg.RANDOM_SEED)
    fold_rows = []

    print(f"{'Fold':>6}  {'Val Acc':>10}  {'Val Loss':>10}  {'Macro F1':>10}")
    print("-" * 44)

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(n_samples), labels), start=1):
        train_loader = make_loader(
            dataset, train_idx.tolist(), args.batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )
        val_loader = make_loader(
            dataset, val_idx.tolist(), args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False,
        )

        model, metrics = train_fold(
            train_loader, val_loader, num_classes,
            epochs=args.epochs, lr=args.lr, device=device,
            no_pretrained=args.no_pretrained,
            variant=args.variant,
        )

        y_true, y_pred = collect_predictions(model, val_loader, device)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        row = {
            "fold": fold,
            "val_accuracy": metrics["val_accuracy"],
            "val_loss": metrics["val_loss"],
            "macro_f1": macro_f1,
        }
        fold_rows.append(row)
        print(
            f"{fold:>6}  {metrics['val_accuracy']*100:>9.2f}%  "
            f"{metrics['val_loss']:>10.4f}  {macro_f1:>10.4f}"
        )

    accs = [r["val_accuracy"] for r in fold_rows]
    f1s = [r["macro_f1"] for r in fold_rows]
    mca_vals = []
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(n_samples), labels)):
        y_t, y_p = all_y_true[fold], all_y_pred[fold]
        mca_vals.append(mean_class_accuracy(y_t, y_p, num_classes))

    print("-" * 44)
    print(f"  Mean Val Acc  : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  Mean Macro F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Mean MCA      : {np.mean(mca_vals)*100:.2f}% ± {np.std(mca_vals)*100:.2f}%")

    # ---- Cross-class misdetection report ------------------------------------
    confusions = find_cross_class_confusions(
        y_true_all, y_pred_all, dataset.classes, top_k=args.top_confusions
    )

    print("\n" + "=" * 62)
    print("  Cross-Class Misdetections (true disease → wrongly detected as)")
    print("=" * 62)
    if confusions:
        for true_cls, pred_cls, count in confusions:
            print(f"  {true_cls:<30} → {pred_cls:<30}  ({count} times)")
    else:
        print("  No cross-class errors in this run (perfect fold predictions).")

    # ---- Save CSV -----------------------------------------------------------
    csv_path = os.path.join(args.logs_dir, f"cross_validation_{args.dataset_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "val_accuracy", "val_loss", "macro_f1"])
        writer.writeheader()
        writer.writerows(fold_rows)
        writer.writerow({
            "fold": "mean",
            "val_accuracy": np.mean(accs),
            "val_loss": np.mean([r["val_loss"] for r in fold_rows]),
            "macro_f1": np.mean(f1s),
        })
        writer.writerow({
            "fold": "std",
            "val_accuracy": np.std(accs),
            "val_loss": np.std([r["val_loss"] for r in fold_rows]),
            "macro_f1": np.std(f1s),
        })
    print(f"\n[cross_validation] Results saved → {csv_path}")

    report_path = os.path.join(
        args.logs_dir, f"cross_class_confusions_{args.dataset_name}.txt"
    )
    with open(report_path, "w") as f:
        f.write(f"Cross-Class Misdetections — {args.dataset_name}\n")
        f.write("=" * 60 + "\n")
        for true_cls, pred_cls, count in confusions:
            f.write(f"{true_cls} → {pred_cls}: {count}\n")
    print(f"[cross_validation] Confusion pairs saved → {report_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified k-fold CV + cross-class misdetection analysis.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Demo",
        choices=list(cfg.DATASET_CLI_CHOICES),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Alias for --dataset_name (e.g. sampletest_plantvillage, realistic).",
    )
    parser.add_argument("--data_dir", type=str, default=cfg.DATA_DIR)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--logs_dir", type=str, default=cfg.LOGS_DIR)
    parser.add_argument("--top_confusions", type=int, default=5)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument(
        "--variant",
        type=str,
        default=cfg.MODEL_VARIANT,
        help="Architecture variant (default: baseline).",
    )
    args = parser.parse_args()
    if args.dataset:
        args.dataset_name = args.dataset
    args.dataset_name = cfg.resolve_dataset_name(args.dataset_name)
    return args


if __name__ == "__main__":
    main(parse_args())
