"""
ablation_study.py — Run all four ConvGeM-NeXt architecture experiments.

Usage
-----
  python ablation_study.py --dataset plantvillage
  python ablation_study.py --dataset plantdoc
  python ablation_study.py --dataset demo --epochs 1 --no_pretrained --batch_size 8
  python ablation_study.py --dataset sampletest_plantvillage --epochs 1 --no_pretrained --batch_size 8
  python ablation_study.py --dataset realistic --epochs 12 --batch_size 32

Each variant is trained and evaluated with the same split, seed, and
hyperparameters. Results are written under results/<variant>/ and
summarised in:

  results/ablation_summary.csv
  results/ablation_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from evaluate import run_evaluation
from model import SUPPORTED_VARIANTS
from train import run_training


def plot_ablation_comparison(rows: list, save_path: str) -> None:
    """Grouped bar chart of Accuracy and Macro F1 using real metrics only."""
    variants = [r["Variant"] for r in rows]
    acc = [float(r["Accuracy"]) * 100.0 for r in rows]
    f1 = [float(r["Macro_F1"]) * 100.0 for r in rows]

    x = np.arange(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_a = ax.bar(x - width / 2, acc, width, label="Accuracy (%)")
    bars_b = ax.bar(x + width / 2, f1, width, label="Macro F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylabel("Score (%)")
    ax.set_title("Architectural Ablation — Accuracy vs Macro F1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(100.0, max(acc + f1 + [1.0]) * 1.15))

    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[ablation] Comparison plot saved → {save_path}")


def write_summary_csv(rows: list, save_path: str) -> None:
    fieldnames = [
        "Variant",
        "Accuracy",
        "Macro_F1",
        "Weighted_F1",
        "Precision",
        "Recall",
        "Parameters",
        "Best_Epoch",
        "Training_Time",
        "Inference_ms",
    ]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ablation] Summary CSV saved → {save_path}")


def run_ablation(args: argparse.Namespace) -> list:
    dataset_name = cfg.resolve_dataset_name(args.dataset)
    results_dir = args.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("  ConvGeM-NeXt Architectural Ablation Study")
    print("=" * 64)
    print(f"  Dataset : {dataset_name}")
    print(f"  Epochs  : {args.epochs}   (same for every variant)")
    print(f"  Batch   : {args.batch_size}")
    print(f"  LR      : {args.lr}")
    print(f"  Seed    : {cfg.RANDOM_SEED}")
    print(f"  Variants: {', '.join(SUPPORTED_VARIANTS)}")
    print("=" * 64 + "\n")

    rows = []
    for variant in SUPPORTED_VARIANTS:
        output_dir = os.path.join(results_dir, variant)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        train_args = SimpleNamespace(
            data_dir=args.data_dir,
            dataset_name=dataset_name,
            dataset=dataset_name,
            variant=variant,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=output_dir,
            logs_dir=output_dir,
            results_dir=results_dir,
            output_dir=output_dir,
            no_pretrained=args.no_pretrained,
            debug=args.debug,
        )

        print(f"\n>>> Training variant: {variant}")
        t0 = time.time()
        train_summary = run_training(train_args)
        train_elapsed = time.time() - t0

        eval_args = SimpleNamespace(
            dataset_name=dataset_name,
            dataset=dataset_name,
            variant=variant,
            data_dir=args.data_dir,
            checkpoint=os.path.join(output_dir, "best_model.pth"),
            batch_size=args.batch_size,
            logs_dir=output_dir,
            results_dir=results_dir,
            output_dir=output_dir,
            no_pretrained=args.no_pretrained,
        )
        print(f"\n>>> Evaluating variant: {variant}")
        metrics = run_evaluation(eval_args)

        row = {
            "Variant": variant,
            "Accuracy": metrics["accuracy"],
            "Macro_F1": metrics["macro_f1"],
            "Weighted_F1": metrics["weighted_f1"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Parameters": metrics.get("parameters", train_summary["parameters"]),
            "Best_Epoch": metrics.get("best_epoch", train_summary["best_epoch"]),
            "Training_Time": train_summary.get("training_time_sec", train_elapsed),
            "Inference_ms": train_summary.get("inference_ms_per_batch", ""),
        }
        rows.append(row)

        combined = {**train_summary, **metrics, "ablation_row": row}
        # train_summary may contain non-JSON-friendly leftovers; write the row + metrics
        with open(os.path.join(output_dir, "ablation_row.json"), "w") as f:
            json.dump(row, f, indent=2)

    write_summary_csv(rows, os.path.join(results_dir, "ablation_summary.csv"))
    plot_ablation_comparison(rows, os.path.join(results_dir, "ablation_comparison.png"))

    print("\n" + "=" * 64)
    print("  Ablation complete (metrics come from this run only)")
    print("=" * 64)
    print(f"{'Variant':<24} {'Acc':>8} {'MacroF1':>10} {'Params':>14} {'BestEp':>8}")
    for r in rows:
        print(
            f"{r['Variant']:<24} {r['Accuracy']*100:7.2f}% "
            f"{r['Macro_F1']:10.4f} {int(r['Parameters']):14,} {int(r['Best_Epoch']):8d}"
        )
    print(f"\nSummary : {os.path.join(results_dir, 'ablation_summary.csv')}")
    print(f"Plot    : {os.path.join(results_dir, 'ablation_comparison.png')}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate all four ConvGeM-NeXt ablation variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=cfg.DATASET,
        help="plantvillage | plantdoc | demo | sampletest_plantvillage | "
             "sampletest_plantdoc | realistic "
             "(canonical SampleTest_* and RealisticPV names also work).",
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="Alias for --dataset.")
    parser.add_argument("--data_dir", type=str, default=cfg.DATA_DIR)
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--results_dir", type=str, default=cfg.RESULTS_DIR)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.dataset_name:
        args.dataset = args.dataset_name
    return args


if __name__ == "__main__":
    run_ablation(parse_args())
