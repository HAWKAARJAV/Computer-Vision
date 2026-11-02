"""
generate_plantdoc_ablation_illustrative.py — Demo PlantDoc ablation folder for teaching.

Mirrors the layout written by ablation_study.py (results/<variant>/, summary CSV,
comparison plot). Baseline accuracy matches the paper (94.69 %); later variants add
small gains derived from our lab ablation gaps. Not from a real PlantDoc run.
Every PNG carries an ILLUSTRATIVE banner.

Output: teaching_material_PlantDoc/ablation_ILLUSTRATIVE/

Usage:
    python generate_plantdoc_ablation_illustrative.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cross_corpus import PD_TO_PV_CLASS_MAP
from model import SUPPORTED_VARIANTS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "teaching_material_PlantDoc" / "ablation_ILLUSTRATIVE"
REAL_RESULTS = ROOT / "results"
ILLUSTRATIVE_SEED = 20261103

BANNER = "ILLUSTRATIVE DEMO — not from a real PlantDoc training run"
SUBTITLE = "Teaching demo only · replace with `ablation_study.py --dataset plantdoc` for measured metrics"

# Paper Table 6 — ConvGeM-NeXt (baseline variant) on PlantDoc
PAPER_ACC = 0.9469
PAPER_MACRO_F1 = 0.9465
PAPER_PRECISION = 0.9477
# Shrink measured lab ablation gaps so scores stay realistic above a 94.69 % baseline
GAP_SCALE = 0.35

METRIC_KEYS = ("Accuracy", "Macro_F1", "Weighted_F1", "Precision", "Recall")
META_KEYS = ("Parameters", "Best_Epoch", "Training_Time", "Inference_ms")


def load_lab_ablation_rows() -> dict[str, dict[str, float]]:
    path = REAL_RESULTS / "ablation_summary.csv"
    rows: dict[str, dict[str, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v = row["Variant"]
            rows[v] = {
                "Accuracy": float(row["Accuracy"]),
                "Macro_F1": float(row["Macro_F1"]),
                "Weighted_F1": float(row["Weighted_F1"]),
                "Precision": float(row["Precision"]),
                "Recall": float(row["Recall"]),
                "Parameters": int(float(row["Parameters"])),
                "Best_Epoch": int(float(row["Best_Epoch"])),
                "Training_Time": float(row["Training_Time"]),
                "Inference_ms": float(row["Inference_ms"]),
            }
    return rows


def build_target_rows() -> dict[str, dict]:
    lab = load_lab_ablation_rows()
    ref = lab["baseline"]
    paper_base = {
        "Accuracy": PAPER_ACC,
        "Macro_F1": PAPER_MACRO_F1,
        "Weighted_F1": PAPER_ACC - 0.006,  # paper does not report; small lag vs acc
        "Precision": PAPER_PRECISION,
        "Recall": 0.8704,  # paper MCA (Table 6 / §4.4.1)
    }
    targets: dict[str, dict] = {}
    for variant in SUPPORTED_VARIANTS:
        scaled = {}
        for key in METRIC_KEYS:
            delta = lab[variant][key] - ref[key]
            val = paper_base[key] + GAP_SCALE * delta
            scaled[key] = float(np.clip(val, 0.0, 0.999))
        for key in META_KEYS:
            scaled[key] = lab[variant][key]
        targets[variant] = scaled
    return targets

SUMMARY_FIELDS = [
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
    "Note",
]

PD_CLASSES = sorted(c for c in PD_TO_PV_CLASS_MAP if c != "Tomato two spotted spider mites leaf")


def banner(fig) -> None:
    fig.text(
        0.5,
        0.995,
        BANNER,
        ha="center",
        va="top",
        fontsize=10,
        color="#b00020",
        fontweight="bold",
    )
    fig.text(0.5, 0.005, SUBTITLE, ha="center", va="bottom", fontsize=8, color="#555")


def save_fig(fig, path: Path) -> None:
    banner(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_template_history(variant: str) -> np.ndarray | None:
    path = REAL_RESULTS / variant / "history.csv"
    if not path.is_file():
        return None
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                [
                    int(row["epoch"]),
                    float(row["train_loss"]),
                    float(row["train_accuracy"]),
                    float(row["val_loss"]),
                    float(row["val_accuracy"]),
                    float(row["lr"]),
                ]
            )
    return np.array(rows, dtype=float)


def synthesize_history(
    target_acc: float,
    best_epoch: int,
    variant: str,
    rng: np.random.Generator,
) -> np.ndarray:
    tpl = load_template_history(variant)
    if tpl is None:
        n = 20
        ep = np.arange(1, n + 1, dtype=float)
        t = 1 - np.exp(-ep / 5.5)
        val_acc = np.clip(0.35 + (target_acc - 0.35) * t / t[-1] + rng.normal(0, 0.012, n), 0, 0.995)
        train_acc = np.clip(val_acc + 0.04 + rng.normal(0, 0.008, n), 0, 0.998)
        train_loss = np.clip(1.2 * np.exp(-ep / 4.5) + 0.1 + rng.normal(0, 0.02, n), 0.05, None)
        val_loss = np.clip(1.0 * np.exp(-ep / 5.0) + 0.2 + rng.normal(0, 0.025, n), 0.08, None)
        lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * ep / n))
        return np.column_stack([ep, train_loss, train_acc, val_loss, val_acc, lr])

    ref_peak = float(np.max(tpl[:, 4]))
    scale = target_acc / max(ref_peak, 1e-6)
    out = tpl.copy()
    out[:, 2] = np.clip(tpl[:, 2] * scale + rng.normal(0, 0.008, len(tpl)), 0, 0.998)
    out[:, 4] = np.clip(tpl[:, 4] * scale + rng.normal(0, 0.012, len(tpl)), 0, 0.995)
    out[:, 1] = np.clip(tpl[:, 1] * (0.95 + rng.uniform(-0.05, 0.05)), 0.05, None)
    out[:, 3] = np.clip(tpl[:, 3] * (0.95 + rng.uniform(-0.05, 0.05)), 0.08, None)
    # Nudge best epoch val acc toward target
    best_i = int(best_epoch) - 1
    best_i = min(max(best_i, 0), len(out) - 1)
    out[best_i, 4] = target_acc
    out[-1, 4] = target_acc - rng.uniform(0.005, 0.02)
    return out


def write_history(path: Path, hist: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"])
        for row in hist:
            w.writerow([int(row[0])] + [f"{v:.6f}" for v in row[1:]])


def plot_loss(hist: np.ndarray, variant: str, path: Path) -> None:
    ep = hist[:, 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, hist[:, 1], label="Train Loss", linewidth=2, marker="o", markersize=4)
    ax.plot(ep, hist[:, 3], label="Val Loss", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title(f"ConvGeM-NeXt [{variant}] — Train vs Val Loss (PlantDoc demo)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    save_fig(fig, path)


def plot_acc(hist: np.ndarray, variant: str, path: Path) -> None:
    ep = hist[:, 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, hist[:, 2] * 100, label="Train Acc", linewidth=2, marker="o", markersize=4)
    ax.plot(ep, hist[:, 4] * 100, label="Val Acc", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"ConvGeM-NeXt [{variant}] — Train vs Val Accuracy (PlantDoc demo)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    save_fig(fig, path)


def synthetic_confusion(rng: np.random.Generator, accuracy: float) -> np.ndarray:
    n = len(PD_CLASSES)
    cm = np.zeros((n, n), dtype=int)
    per_class = rng.uniform(accuracy - 0.12, accuracy + 0.06, n)
    per_class = np.clip(per_class, 0.55, 0.99)
    n_test = 236
    counts = rng.multinomial(n_test, np.ones(n) / n)
    for i in range(n):
        correct = int(round(counts[i] * per_class[i]))
        cm[i, i] = correct
        wrong = counts[i] - correct
        if wrong > 0:
            js = [j for j in range(n) if j != i]
            probs = rng.dirichlet(np.ones(len(js)))
            splits = (probs * wrong).astype(int)
            splits[-1] += wrong - splits.sum()
            for j, c in zip(js, splits):
                cm[i, j] += c
    return cm


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    n = len(PD_CLASSES)
    fig_size = max(10, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=PD_CLASSES,
        yticklabels=PD_CLASSES,
        ax=ax,
        cbar_kws={"shrink": 0.7},
        linewidths=0.3,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — PlantDoc [multiscale_attention] (demo layout)", fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    save_fig(fig, path)


def write_confusion_csv(cm: np.ndarray, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PD_CLASSES)
        for i, row in enumerate(cm):
            w.writerow([PD_CLASSES[i]] + list(row))


def write_metrics(variant: str, row: dict, hist: np.ndarray, out_dir: Path) -> None:
    best_i = int(row["Best_Epoch"]) - 1
    best_i = min(max(best_i, 0), len(hist) - 1)
    payload = {
        "variant": variant,
        "dataset_name": "PlantDoc",
        "num_classes": 27,
        "note": BANNER,
        "accuracy": row["Accuracy"],
        "macro_f1": row["Macro_F1"],
        "weighted_f1": row["Weighted_F1"],
        "precision": row["Precision"],
        "recall": row["Recall"],
        "parameters": row["Parameters"],
        "best_epoch": int(row["Best_Epoch"]),
        "best_val_accuracy": float(hist[best_i, 4]),
        "training_time_sec": row["Training_Time"],
        "inference_ms_per_batch": row["Inference_ms"],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


def plot_ablation_comparison(rows: list[dict], path: Path) -> None:
    variants = [r["Variant"] for r in rows]
    acc = [float(r["Accuracy"]) * 100 for r in rows]
    f1 = [float(r["Macro_F1"]) * 100 for r in rows]
    x = np.arange(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_a = ax.bar(x - width / 2, acc, width, label="Accuracy (%)")
    bars_b = ax.bar(x + width / 2, f1, width, label="Macro F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylabel("Score (%)")
    ax.set_title("Architectural Ablation on PlantDoc — demo metrics (ILLUSTRATIVE)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(75, 100)
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
    save_fig(fig, path)


def main() -> None:
    rng = np.random.default_rng(ILLUSTRATIVE_SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    target_rows = build_target_rows()

    summary_rows = []
    for variant in SUPPORTED_VARIANTS:
        row = target_rows[variant]
        out_dir = OUT / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        hist = synthesize_history(row["Accuracy"], row["Best_Epoch"], variant, rng)
        write_history(out_dir / "history.csv", hist)
        plot_loss(hist, variant, out_dir / "loss_curve.png")
        plot_acc(hist, variant, out_dir / "accuracy_curve.png")
        write_metrics(variant, row, hist, out_dir)

        if variant == "multiscale_attention":
            cm = synthetic_confusion(rng, row["Accuracy"])
            write_confusion_csv(cm, out_dir / "confusion_matrix.csv")
            plot_confusion(cm, out_dir / "confusion_matrix.png")

        summary_rows.append(
            {
                "Variant": variant,
                **{
                    k: (f"{row[k]:.4f}" if k in METRIC_KEYS else row[k])
                    for k in SUMMARY_FIELDS
                    if k in row
                },
                "Note": "ILLUSTRATIVE_DEMO",
            }
        )

    with open(OUT / "ablation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)

    plot_ablation_comparison(
        [{**r, **{k: float(r[k]) for k in METRIC_KEYS if k in r}} for r in summary_rows],
        OUT / "ablation_comparison.png",
    )

    acc_line = " → ".join(
        f"{v}: {target_rows[v]['Accuracy'] * 100:.2f} %"
        for v in SUPPORTED_VARIANTS
    )
    readme = OUT / "README.md"
    readme.write_text(
        "# PlantDoc ablation — ILLUSTRATIVE demo\n\n"
        f"**{BANNER}**\n\n"
        "Same folder layout as `results/` from `ablation_study.py`, for classroom walkthroughs "
        "when a full PlantDoc ablation has not finished yet. Baseline test accuracy matches "
        f"the paper (**94.69 %**); each next variant adds a small gain from our lab ablation "
        f"gaps (scaled): {acc_line}.\n\n"
        "Regenerate:\n\n"
        "```bash\n"
        "python generate_plantdoc_ablation_illustrative.py\n"
        "```\n\n"
        "For graded work, run `python ablation_study.py --dataset plantdoc` and use those files instead.\n",
        encoding="utf-8",
    )
    print(f"Illustrative PlantDoc ablation written to {OUT}")


if __name__ == "__main__":
    main()
