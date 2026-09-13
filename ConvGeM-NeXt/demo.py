"""
demo.py — One-command presentation demo (no large dataset download).

Shows your teacher:
  1. Synthetic mini dataset creation
  2. ConvGeM-NeXt architecture + forward pass
  3. Quick training run (2 epochs on CPU-friendly settings)
  4. Evaluation metrics + sample predictions

Run:
    cd ConvGeM-NeXt
    python3 demo.py

Skip training (architecture + data pipeline only, ~30 seconds):
    python3 demo.py --no-train
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import config as cfg
from create_demo_dataset import create_demo_dataset, DEMO_CLASSES
from dataset import get_dataloaders
from model import ConvGeMNeXt


def print_banner(title: str) -> None:
    line = "=" * 62
    print(f"\n{line}\n  {title}\n{line}")


def show_paper_results() -> None:
    print("\nPaper results (full PlantVillage / PlantDoc datasets):")
    print("  PlantVillage — Top-1: 99.65%  |  Macro F1: 0.9959  (38 classes)")
    print("  PlantDoc     — Top-1: 94.69%  |  Macro F1: 0.9465  (27 classes)")
    print("  (Demo uses 5 synthetic classes — metrics are for pipeline proof only.)\n")


def run_forward_demo(model: ConvGeMNeXt, device: torch.device) -> None:
    dummy = torch.randn(2, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(dummy)
    print(f"  Input batch  : {tuple(dummy.shape)}")
    print(f"  Output logits: {tuple(logits.shape)}")
    print(f"  GeM exponent p: {model.gem_pool.p.item():.4f}")


def sample_predictions(
    model: ConvGeMNeXt,
    loader,
    class_names: list,
    device: torch.device,
    n: int = 5,
) -> None:
    model.eval()
    shown = 0
    print("\nSample predictions on test images:")
    print("-" * 62)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            for i in range(images.size(0)):
                if shown >= n:
                    return
                true_name = class_names[labels[i].item()]
                pred_name = class_names[preds[i].item()]
                conf = probs[i, preds[i]].item() * 100
                mark = "✓" if preds[i] == labels[i] else "✗"
                print(f"  {mark} True: {true_name:<28} → Pred: {pred_name:<28} ({conf:.1f}%)")
                shown += 1


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_banner("ConvGeM-NeXt — Live Demo (no dataset download)")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    else:
        print("  Note   : Running on CPU — demo uses 2 epochs + small data.")

    # ---- Step 1: synthetic data -------------------------------------------
    print_banner("Step 1 — Create tiny demo dataset")
    create_demo_dataset(
        data_dir=args.data_dir,
        dataset_name="Demo",
        images_per_class=args.images_per_class,
        force=args.force,
    )
    print(f"  Classes: {', '.join(DEMO_CLASSES)}")

    # ---- Step 2: data pipeline --------------------------------------------
    print_banner("Step 2 — Load data (70/15/15 stratified split)")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(
        data_dir=args.data_dir,
        dataset_name="Demo",
        batch_size=args.batch_size,
        img_size=cfg.IMG_SIZE,
        num_workers=0,
        pin_memory=False,
        random_seed=cfg.RANDOM_SEED,
    )
    images, labels = next(iter(train_loader))
    print(f"  Train batch shape: {tuple(images.shape)}")

    # ---- Step 3: model ----------------------------------------------------
    print_banner("Step 3 — ConvGeM-NeXt architecture")
    num_classes = len(dataset.classes)
    model = ConvGeMNeXt(
        num_classes=num_classes,
        variant="baseline",
        p_init=cfg.GEM_P_INIT,
        dropout=cfg.DROPOUT,
        pretrained=args.pretrained,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Backbone   : ConvNeXt-Base → GeM Pooling → Custom Head")
    if not args.pretrained:
        print("  Weights    : random init (offline demo — no ImageNet download)")
    run_forward_demo(model, device)

    show_paper_results()

    if args.no_train:
        print_banner("Demo complete (pipeline verified, training skipped)")
        print("  To run a short training demo:  python3 demo.py")
        print("  Full training later:           python3 train.py --dataset_name PlantVillage")
        return

    # ---- Step 4: quick training -------------------------------------------
    print_banner(f"Step 4 — Quick training ({args.epochs} epoch(s) on Demo data)")
    print("  (This proves the full train → save → eval pipeline works.)\n")

    import subprocess
    train_cmd = [
        sys.executable, "train.py",
        "--data_dir", args.data_dir,
        "--dataset_name", "Demo",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--checkpoint_dir", args.checkpoint_dir,
        "--logs_dir", args.logs_dir,
        "--no_pretrained",
    ]
    t0 = time.time()
    result = subprocess.run(train_cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print("\nTraining failed. Try:  python3 demo.py --no-train")
        sys.exit(result.returncode)
    print(f"\n  Training finished in {time.time() - t0:.0f}s")

    # ---- Step 5: evaluation -----------------------------------------------
    print_banner("Step 5 — Evaluate on test split")
    eval_cmd = [
        sys.executable, "evaluate.py",
        "--dataset_name", "Demo",
        "--data_dir", args.data_dir,
        "--checkpoint", os.path.join(args.checkpoint_dir, "best_model.pth"),
        "--batch_size", str(args.batch_size),
        "--logs_dir", args.logs_dir,
        "--no_pretrained",
    ]
    subprocess.run(eval_cmd, cwd=Path(__file__).parent, check=True)

    # ---- Step 6: sample predictions ---------------------------------------
    print_banner("Step 6 — Sample predictions")
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    sample_predictions(model, test_loader, dataset.classes, device)

    print_banner("Demo complete — ready to show your teacher")
    print("  GitHub  : https://github.com/HAWKAARJAV/ConvGeM-NeXt")
    print("  Outputs : checkpoints/best_model.pth")
    print("            logs/loss_curve_Demo.png")
    print("            logs/confusion_matrix_Demo.png")
    print("\n  Explain: full paper uses PlantVillage (38 cls) / PlantDoc (27 cls).")
    print("           This demo uses 5 synthetic classes to prove the pipeline.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConvGeM-NeXt presentation demo.")
    parser.add_argument("--data_dir", type=str, default=cfg.DATA_DIR)
    parser.add_argument("--images_per_class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2, help="Quick training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Small batch for CPU demo.")
    parser.add_argument("--checkpoint_dir", type=str, default=cfg.CHECKPOINT_DIR)
    parser.add_argument("--logs_dir", type=str, default=cfg.LOGS_DIR)
    parser.add_argument("--no-train", action="store_true", help="Skip training (fastest).")
    parser.add_argument("--pretrained", action="store_true", help="Download ImageNet weights (needs internet).")
    parser.add_argument("--force", action="store_true", help="Regenerate demo images.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
