"""
train.py — Training Loop for ConvGeM-NeXt
==========================================
Features:
  * AdamW optimiser with CosineAnnealingLR scheduler
  * Mixed-precision training via torch.cuda.amp  (AMP)
  * Best-checkpoint saving (lowest validation loss)
  * Epoch-level CSV logging (epoch, train_loss, val_loss, val_acc)
  * Training + validation loss curve plot saved as PNG
  * Argparse CLI for common hyperparameters
  * Estimated time remaining printed each epoch

Usage
-----
  # Train on PlantVillage (default):
  python train.py --data_dir data --dataset_name PlantVillage

  # Override epochs and batch size:
  python train.py --dataset_name PlantDoc --epochs 30 --batch_size 32 --lr 5e-4
"""

import os
import csv
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")           # Non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt

import config as cfg
from model import ConvGeMNeXt
from dataset import get_dataloaders


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
) -> None:
    """Plot training vs validation loss and save as PNG.

    Args:
        train_losses: Per-epoch training losses.
        val_losses:   Per-epoch validation losses.
        save_path:    Destination file path (must end in ``.png``).
    """
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, val_losses,   label="Val Loss",   linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title("ConvGeM-NeXt — Training vs Validation Loss", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[train] Loss curve saved → {save_path}")


# ---------------------------------------------------------------------------
# Core training / validation functions
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean cross-entropy loss.

    Uses Automatic Mixed Precision (AMP):
      * Forward + loss computed in float16 (via autocast) for speed.
      * Scaler adjusts gradients to avoid underflow before backward pass.

    Args:
        model:     ConvGeMNeXt in training mode.
        loader:    Training DataLoader.
        criterion: CrossEntropyLoss.
        optimizer: AdamW.
        scaler:    GradScaler for AMP.
        device:    Target compute device.

    Returns:
        Average loss over all mini-batches.
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        # Move data to GPU (non-blocking because DataLoader uses pin_memory)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

        # ---- Forward pass under autocast (float16 matmuls) -----------------
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        # ---- Backward pass with scaled gradients ---------------------------
        scaler.scale(loss).backward()

        # Optional gradient clipping — prevents exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model on a data split.

    Args:
        model:     ConvGeMNeXt in eval mode.
        loader:    Val or test DataLoader.
        criterion: CrossEntropyLoss.
        device:    Target compute device.

    Returns:
        Tuple ``(avg_loss, top1_accuracy)`` — both as Python floats.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # AMP autocast during inference too — free speed improvement
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Full training pipeline: setup → train → save best checkpoint → plot."""

    # ---- Device setup ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print(f"  ConvGeM-NeXt Training")
    print(f"  Device   : {device}")
    if device.type == "cuda":
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # auto-tune CUDA kernels for fixed input size
    else:
        print(
            "  WARNING: CUDA not available. Training on CPU will be VERY slow.\n"
            "           Consider using a machine with a GPU or Google Colab."
        )
    print(f"  Dataset  : {args.dataset_name}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch sz : {args.batch_size}")
    print(f"  LR       : {args.lr}")
    print("="*60 + "\n")

    # ---- Create output directories -----------------------------------------
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    # ---- Data loaders -------------------------------------------------------
    num_workers = cfg.NUM_WORKERS if device.type == "cuda" else 0
    pin_memory  = cfg.PIN_MEMORY  if device.type == "cuda" else False

    train_loader, val_loader, _, dataset = get_dataloaders(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        img_size=cfg.IMG_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
        random_seed=cfg.RANDOM_SEED,
    )

    # ---- Model -------------------------------------------------------------
    num_classes = cfg.get_num_classes(args.dataset_name)
    model = ConvGeMNeXt(
        num_classes=num_classes,
        p_init=cfg.GEM_P_INIT,
        dropout=cfg.DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # ---- Loss, optimiser, scheduler ----------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # Cosine annealing: LR decays from args.lr → ~0 over T_MAX steps
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # GradScaler for AMP — does nothing (scale=1) on CPU
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ---- CSV logging setup -------------------------------------------------
    log_path = os.path.join(args.logs_dir, f"training_log_{args.dataset_name}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "lr"])

    # ---- Training loop -----------------------------------------------------
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")

    train_losses: list = []
    val_losses:   list = []

    epoch_times: list = []   # track wall-clock time per epoch for ETA

    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val Acc':>9}  {'LR':>10}  {'ETA':>10}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        # -- One epoch of training -------------------------------------------
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # -- Validation pass -------------------------------------------------
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # -- LR step (after optimiser step) ----------------------------------
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # -- Book-keeping ----------------------------------------------------
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_dur = time.time() - t_start
        epoch_times.append(epoch_dur)
        remaining_epochs = args.epochs - epoch
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        eta = format_seconds(avg_epoch_time * remaining_epochs)

        # Print progress
        print(
            f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}  "
            f"{val_acc*100:>8.2f}%  {current_lr:>10.2e}  {eta:>10}"
        )

        # -- CSV log ----------------------------------------------------------
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_acc, current_lr])

        # -- Save best checkpoint --------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "num_classes": num_classes,
                    "dataset_name": args.dataset_name,
                },
                checkpoint_path,
            )
            print(f"         ✓ Best model saved  (val_loss={val_loss:.4f})")

    print("-" * 70)
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"CSV log    : {log_path}")

    # ---- Save loss curve plot ----------------------------------------------
    plot_path = os.path.join(args.logs_dir, f"loss_curve_{args.dataset_name}.png")
    save_loss_plot(train_losses, val_losses, save_path=plot_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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
        default=cfg.DATASET_NAME,
        choices=["PlantVillage", "PlantDoc"],
        help="Which dataset to train on.",
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
        default=cfg.CHECKPOINT_DIR,
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=cfg.LOGS_DIR,
        help="Directory to save CSV logs and plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
