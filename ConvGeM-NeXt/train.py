"""
train.py — Training Loop for ConvGeM-NeXt
==========================================
Features:
  * Ablation variants: baseline | multiscale | attention | multiscale_attention
  * AdamW optimiser with CosineAnnealingLR scheduler
  * Mixed-precision training via torch.cuda.amp  (AMP)
  * Best-checkpoint saving (lowest validation loss)
  * Per-variant results folder (metrics, history, config, plots)
  * Argparse CLI for common hyperparameters

Usage
-----
  python train.py --dataset_name PlantVillage --variant baseline
  python train.py --dataset plantvillage --variant multiscale
  python train.py --dataset_name PlantDoc --variant multiscale_attention
  python train.py --dataset sampletest_plantvillage --epochs 2 --batch_size 8 --no_pretrained
  python train.py --dataset sampletest_plantdoc --epochs 2 --batch_size 8 --no_pretrained
"""

import json
import os
import csv
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from model import ConvGeMNeXt, SUPPORTED_VARIANTS, count_parameters
from dataset import get_dataloaders


def set_seed(seed: int) -> None:
    """Fix Python / NumPy / PyTorch RNGs for comparable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_seconds(seconds: float) -> str:
    """Convert a float number of seconds to a human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    elif m:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def save_loss_plot(
    train_losses: list,
    val_losses: list,
    save_path: str,
    title: str = "ConvGeM-NeXt — Training vs Validation Loss",
) -> None:
    """Plot training vs validation loss and save as PNG."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, val_losses,   label="Val Loss",   linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[train] Loss curve saved → {save_path}")


def save_acc_plot(
    train_accs: list,
    val_accs: list,
    save_path: str,
    title: str = "ConvGeM-NeXt — Training vs Validation Accuracy",
) -> None:
    epochs = range(1, len(train_accs) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [a * 100 for a in train_accs], label="Train Acc", linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, [a * 100 for a in val_accs], label="Val Acc", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[train] Accuracy curve saved → {save_path}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple:
    """Run one training epoch and return ``(mean_loss, accuracy)``."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model on a data split. Returns ``(avg_loss, top1_accuracy)``."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def measure_inference_time(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 4,
    warmup: int = 5,
    repeats: int = 20,
) -> float:
    """Return mean milliseconds per dummy batch (eval mode)."""
    model.eval()
    dummy = torch.randn(batch_size, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(sum(times) / len(times))


def run_training(args: argparse.Namespace) -> dict:
    """Full training pipeline. Returns a result dictionary for ablation."""
    set_seed(cfg.RANDOM_SEED)

    dataset_name = cfg.resolve_dataset_name(args.dataset_name)
    variant = cfg.resolve_variant(args.variant)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(args.results_dir, variant)
    checkpoint_dir = args.checkpoint_dir or output_dir
    logs_dir = args.logs_dir or output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ConvGeM-NeXt Training")
    print(f"  Device   : {device}")
    if device.type == "cuda":
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print(
            "  WARNING: CUDA not available. Training on CPU will be VERY slow.\n"
            "           Consider using a machine with a GPU or Google Colab."
        )
    print(f"  Dataset  : {dataset_name}")
    print(f"  Variant  : {variant}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch sz : {args.batch_size}")
    print(f"  LR       : {args.lr}")
    print(f"  Output   : {output_dir}")
    print("=" * 60 + "\n")

    num_workers = cfg.NUM_WORKERS if device.type == "cuda" else 0
    pin_memory = cfg.PIN_MEMORY if device.type == "cuda" else False

    train_loader, val_loader, test_loader, dataset = get_dataloaders(
        data_dir=args.data_dir,
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        img_size=cfg.IMG_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
        random_seed=cfg.RANDOM_SEED,
    )

    num_classes = len(dataset.classes)
    model = ConvGeMNeXt(
        num_classes=num_classes,
        variant=variant,
        p_init=cfg.GEM_P_INIT,
        dropout=cfg.DROPOUT,
        pretrained=not args.no_pretrained,
        debug=args.debug,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}  (trainable {trainable_params:,})\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    run_config = {
        "variant": variant,
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": cfg.WEIGHT_DECAY,
        "img_size": cfg.IMG_SIZE,
        "mean": cfg.MEAN,
        "std": cfg.STD,
        "random_seed": cfg.RANDOM_SEED,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropyLoss",
        "dropout": cfg.DROPOUT,
        "gem_p_init": cfg.GEM_P_INIT,
        "pretrained": not args.no_pretrained,
        "parameters": total_params,
        "trainable_parameters": trainable_params,
    }
    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    log_path = os.path.join(logs_dir, f"training_log_{dataset_name}.csv")
    history_path = os.path.join(output_dir, "history.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"])
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"])

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    epoch_times = []
    t_train_start = time.time()

    print(
        f"{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  "
        f"{'Val Loss':>10}  {'Val Acc':>9}  {'LR':>10}  {'ETA':>10}"
    )
    print("-" * 86)

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_dur = time.time() - t_start
        epoch_times.append(epoch_dur)
        remaining_epochs = args.epochs - epoch
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        eta = format_seconds(avg_epoch_time * remaining_epochs)

        print(
            f"{epoch:>6}  {train_loss:>11.4f}  {train_acc*100:>9.2f}%  "
            f"{val_loss:>10.4f}  {val_acc*100:>8.2f}%  "
            f"{current_lr:>10.2e}  {eta:>10}"
        )

        row = [epoch, train_loss, train_acc, val_loss, val_acc, current_lr]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "num_classes": num_classes,
                    "dataset_name": dataset_name,
                    "variant": variant,
                    "no_pretrained": args.no_pretrained,
                    "parameters": total_params,
                },
                checkpoint_path,
            )
            print(f"         ✓ Best model saved  (val_loss={val_loss:.4f})")

    training_time = time.time() - t_train_start
    print("-" * 86)
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"CSV log    : {log_path}")

    plot_path = os.path.join(logs_dir, f"loss_curve_{dataset_name}.png")
    save_loss_plot(
        train_losses, val_losses, save_path=plot_path,
        title=f"ConvGeM-NeXt [{variant}] — Train vs Val Loss",
    )
    acc_plot_path = os.path.join(output_dir, "accuracy_curve.png")
    save_acc_plot(
        train_accs, val_accs, save_path=acc_plot_path,
        title=f"ConvGeM-NeXt [{variant}] — Train vs Val Accuracy",
    )

    # Reload best weights before test evaluation / inference timing
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    infer_ms = measure_inference_time(model, device, batch_size=min(4, args.batch_size))

    summary = {
        "variant": variant,
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_train_accuracy": train_accs[-1] if train_accs else None,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "training_time_sec": training_time,
        "inference_ms_per_batch": infer_ms,
        "checkpoint": checkpoint_path,
        "output_dir": output_dir,
        "run_config": config_path,
    }
    summary_path = os.path.join(output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Test loss / acc (best ckpt): {test_loss:.4f} / {test_acc*100:.2f}%")
    print(f"Training time : {format_seconds(training_time)}")
    print(f"Inference     : {infer_ms:.2f} ms / dummy batch")
    print(f"Summary       : {summary_path}")
    return summary


def main(args: argparse.Namespace) -> None:
    run_training(args)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ConvGeM-NeXt for plant disease classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=cfg.DATA_DIR,
        help="Root directory containing dataset sub-folders.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset folder name (PlantVillage, PlantDoc, Demo, "
             "SampleTest_PlantVillage, SampleTest_PlantDoc, RealisticPV).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Alias for --dataset_name (plantvillage | plantdoc | demo | "
             "sampletest_plantvillage | sampletest_plantdoc | realistic).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=cfg.MODEL_VARIANT,
        help="Architecture variant: baseline | multiscale | attention | multiscale_attention",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg.NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=cfg.BATCH_SIZE,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg.LEARNING_RATE,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (defaults to results/<variant>/).",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=None,
        help="Directory to save CSV logs and plots (defaults to results/<variant>/).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=cfg.RESULTS_DIR,
        help="Root folder for per-variant experiment outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Exact output folder (overrides results_dir/variant).",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Skip ImageNet backbone weights (useful for offline demo runs).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print tensor shapes on the first forward pass.",
    )
    args = parser.parse_args()
    raw_dataset = args.dataset_name or args.dataset or cfg.DATASET_NAME
    args.dataset_name = raw_dataset
    if args.variant not in SUPPORTED_VARIANTS:
        # Allow aliases; resolve later. Unknown values fail in resolve_variant.
        pass
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
