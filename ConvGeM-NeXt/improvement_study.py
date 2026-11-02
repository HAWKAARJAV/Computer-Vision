"""
improvement_study.py — Cumulative study of proposed improvements over ConvGeM-NeXt
==================================================================================
Starts from the paper's recipe (Arshad et al., 2026: no augmentation, AdamW 1e-3,
cosine, 30 epochs, 224 px) and adds one improvement per step, so the gain of each
change is measured on the same split, seed and test set:

  S0  paper            Paper recipe (reproduction)
  S1  +aug             Field-robust augmentation
  S2  +ls              + label smoothing 0.1
  S3  +balanced        + class-balanced sampling
  S4  +disc_lr         + backbone LR ×0.1 and 2 warm-up epochs
  S5  +pv_init         + initialise from a PlantVillage checkpoint (needs --pv_checkpoint)
  S6  +tta             + test-time augmentation (evaluation only; reuses S5/S4 weights)

Every number written by this script comes from an actual training + evaluation
run on your machine. Nothing is simulated.

Usage
-----
  python improvement_study.py --dataset plantdoc --variant baseline \
      --pv_checkpoint results/baseline/best_model.pth

  # Run only some steps (e.g. resume after S2 finished)
  python improvement_study.py --dataset plantdoc --steps S3 S4 S5 S6

  # Pipeline check on synthetic data
  python improvement_study.py --dataset sampletest_plantdoc --epochs 1 --batch_size 8 --no_pretrained

Outputs: results/improvements/<Dataset>/<variant>/
  <step>/                    per-step run (history, metrics.json, confusion matrix …)
  improvement_summary.csv    Step, Description, Accuracy, Macro_F1, Macro_Precision, MCA, Test_Loss, Δ
  improvement_comparison.png Accuracy / Macro-F1 per cumulative step
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from evaluate import run_evaluation
from train import run_training

STEPS = [
    ("S0", "Paper recipe (reproduction)", {}),
    ("S1", "+ Field-robust augmentation", {"augment": True}),
    ("S2", "+ Label smoothing 0.1", {"label_smoothing": 0.1}),
    ("S3", "+ Class-balanced sampling", {"balanced_sampler": True}),
    ("S4", "+ Backbone LR x0.1, 2 warm-up epochs", {"backbone_lr_mult": 0.1, "warmup_epochs": 2}),
    ("S5", "+ PlantVillage initialisation", {"init_checkpoint": "__PV__"}),
    ("S6", "+ Test-time augmentation", {"__tta__": True}),
]


def cumulative_opts(upto: str, pv_checkpoint: str | None) -> dict:
    opts: dict = {}
    for step, _, delta in STEPS:
        opts.update(delta)
        if step == upto:
            break
    if opts.get("init_checkpoint") == "__PV__":
        opts["init_checkpoint"] = pv_checkpoint
    return opts


def run_step(args, step: str, desc: str, vdir: str, dataset: str, variant: str) -> dict | None:
    opts = cumulative_opts(step, args.pv_checkpoint)
    tta = bool(opts.pop("__tta__", False))
    if not tta and "init_checkpoint" in opts and not opts["init_checkpoint"]:
        print(f"[improvements] {step} skipped: pass --pv_checkpoint to use PlantVillage initialisation.")
        return None
    out_dir = os.path.join(vdir, step)

    if tta:
        # TTA changes evaluation only: reuse the latest trained step's weights.
        prev = next((s for s, _, _ in reversed(STEPS[:-1])
                     if os.path.isfile(os.path.join(vdir, s, "best_model.pth"))), None)
        if prev is None:
            print(f"[improvements] {step} skipped: no trained step to evaluate with TTA.")
            return None
        ckpt = os.path.join(vdir, prev, "best_model.pth")
        desc = f"{desc} (weights of {prev})"
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        run_training(SimpleNamespace(
            data_dir=args.data_dir, dataset_name=dataset, variant=variant,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            checkpoint_dir=out_dir, logs_dir=out_dir, results_dir=vdir, output_dir=out_dir,
            no_pretrained=args.no_pretrained, debug=False, **opts,
        ))
        ckpt = os.path.join(out_dir, "best_model.pth")

    metrics = run_evaluation(SimpleNamespace(
        dataset_name=dataset, dataset=dataset, variant=variant, data_dir=args.data_dir,
        checkpoint=ckpt, batch_size=args.batch_size, logs_dir=out_dir,
        results_dir=vdir, output_dir=out_dir, no_pretrained=args.no_pretrained, tta=tta,
    ))
    test_loss = float("nan")
    summary_path = os.path.join(os.path.dirname(ckpt), "train_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            test_loss = float(json.load(f).get("test_loss", float("nan")))
    return {
        "Step": step, "Description": desc,
        "Accuracy": metrics["accuracy"], "Macro_F1": metrics["macro_f1"],
        "Macro_Precision": metrics["precision"], "MCA": metrics["mca"],
        "Test_Loss": test_loss if not tta else float("nan"),
    }


def write_outputs(rows: list, vdir: str, dataset: str, variant: str) -> None:
    rows = sorted(rows, key=lambda r: r["Step"])
    base = next((r["Accuracy"] for r in rows if r["Step"] == "S0"), None)
    csv_path = os.path.join(vdir, "improvement_summary.csv")
    fields = ["Step", "Description", "Accuracy", "Macro_F1", "Macro_Precision", "MCA",
              "Test_Loss", "Delta_Acc_vs_S0"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()},
                        "Delta_Acc_vs_S0": "" if base is None else f"{(r['Accuracy'] - base) * 100:+.2f}"})
    print(f"[improvements] Summary → {csv_path}")

    labels = [r["Step"] for r in rows]
    acc = [r["Accuracy"] * 100 for r in rows]
    f1 = [r["Macro_F1"] * 100 for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ba = ax.bar(x - 0.2, acc, 0.4, label="Accuracy (%)")
    bf = ax.bar(x + 0.2, f1, 0.4, label="Macro F1 (%)")
    for bars, vals in ((ba, acc), (bf, f1)):
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Step']}\n{r['Description'].lstrip('+ ')[:22]}" for r in rows], fontsize=8)
    lo = max(0.0, min(acc + f1) - 10)
    ax.set_ylim(lo, 100)
    ax.set_ylabel("Test score (%)")
    ax.set_title(f"Cumulative improvements over ConvGeM-NeXt — {dataset} [{variant}]")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    png = os.path.join(vdir, "improvement_comparison.png")
    plt.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[improvements] Plot → {png}")


def main(args) -> None:
    dataset = cfg.resolve_dataset_name(args.dataset)
    variant = cfg.resolve_variant(args.variant)
    vdir = os.path.join(args.results_dir, "improvements", dataset, variant)
    Path(vdir).mkdir(parents=True, exist_ok=True)

    rows_path = os.path.join(vdir, "rows.json")
    rows = {}
    if os.path.isfile(rows_path):
        with open(rows_path) as f:
            rows = {r["Step"]: r for r in json.load(f)}

    for step, desc, _ in STEPS:
        if args.steps and step not in args.steps:
            continue
        row = run_step(args, step, desc, vdir, dataset, variant)
        if row is not None:
            rows[step] = row
            with open(rows_path, "w") as f:
                json.dump(list(rows.values()), f, indent=2)
            write_outputs(list(rows.values()), vdir, dataset, variant)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cumulative improvement study for ConvGeM-NeXt.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", type=str, default="plantdoc")
    p.add_argument("--variant", type=str, default="baseline")
    p.add_argument("--pv_checkpoint", type=str, default=None,
                   help="PlantVillage best_model.pth for step S5 (same variant).")
    p.add_argument("--steps", nargs="*", default=None, help="Subset of steps to run, e.g. S0 S1.")
    p.add_argument("--data_dir", type=str, default=cfg.DATA_DIR)
    p.add_argument("--results_dir", type=str, default=cfg.RESULTS_DIR)
    p.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--no_pretrained", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
