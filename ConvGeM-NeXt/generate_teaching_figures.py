"""
generate_teaching_figures.py — Synthetic ablation plots for classroom demos.

Does NOT read or overwrite results/ from real training runs.
Output lives under teaching_materials/ only.

Usage:
    python generate_teaching_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import SUPPORTED_VARIANTS

# Fixed seed — documented in teaching_materials/README.md
ILLUSTRATIVE_SEED = 20260417

OUTPUT_DIR = Path(__file__).resolve().parent / "teaching_materials"
N_EPOCHS = 20

TITLE_SUFFIX = "ILLUSTRATIVE — not measured from your run"
SUBTITLE = "Illustrative example — not measured from your run"

# Final validation accuracy targets (fractions) — pedagogical ordering only
VARIANT_TARGETS = {
    "baseline": {"val_acc": 0.889, "macro_f1": 0.876, "best_epoch": 16},
    "multiscale": {"val_acc": 0.918, "macro_f1": 0.907, "best_epoch": 17},
    "attention": {"val_acc": 0.931, "macro_f1": 0.921, "best_epoch": 15},
    "multiscale_attention": {"val_acc": 0.951, "macro_f1": 0.943, "best_epoch": 18},
}

# Parameter counts match real ConvGeM-NeXt variant sizes (structure only, not run metrics)
VARIANT_PARAMS = {
    "baseline": 88_655_015,
    "multiscale": 90_097_831,
    "attention": 88_787_274,
    "multiscale_attention": 90_106_394,
}

INFERENCE_MS = {
    "baseline": 31.2,
    "multiscale": 98.5,
    "attention": 32.1,
    "multiscale_attention": 34.8,
}


def _sigmoid_ramp(epochs: np.ndarray, start: float, end: float, steepness: float, midpoint: float) -> np.ndarray:
    t = 1.0 / (1.0 + np.exp(-steepness * (epochs - midpoint)))
    t = (t - t[0]) / (t[-1] - t[0] + 1e-9)
    return start + (end - start) * t


def synthesize_curves(
    rng: np.random.Generator,
    final_val_acc: float,
    final_train_acc: float | None = None,
    n_epochs: int = N_EPOCHS,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Train smoother; val improves with occasional epoch-to-epoch dips."""
    if final_train_acc is None:
        final_train_acc = min(0.985, final_val_acc + 0.028)

    ep = np.arange(1, n_epochs + 1, dtype=float)
    val_base = _sigmoid_ramp(ep, 0.32, final_val_acc, 0.55, 8.5)
    train_base = _sigmoid_ramp(ep, 0.38, final_train_acc, 0.62, 7.0)

    val_noise = rng.normal(0, 0.012, n_epochs)
    train_noise = rng.normal(0, 0.006, n_epochs)

    # A few deliberate validation bumps (common when val loss plateaus)
    bump_epochs = rng.choice(np.arange(6, n_epochs - 2), size=3, replace=False)
    for be in bump_epochs:
        val_noise[be] -= rng.uniform(0.015, 0.035)

    val_acc = np.clip(val_base + val_noise, 0.25, 0.995)
    train_acc = np.clip(train_base + train_noise, 0.30, 0.998)

    # Monotonic-ish trend: val should not end below mid-run best by much
    running_max = np.maximum.accumulate(val_acc)
    val_acc = 0.55 * val_acc + 0.45 * running_max

    # Loss inversely related to accuracy (illustrative CE scale)
    val_loss = 2.4 * np.exp(-2.8 * val_acc) + rng.normal(0, 0.02, n_epochs)
    train_loss = 2.0 * np.exp(-3.2 * train_acc) + rng.normal(0, 0.012, n_epochs)
    val_loss = np.clip(val_loss, 0.08, 2.8)
    train_loss = np.clip(train_loss, 0.05, 2.5)

    return (
        train_loss.tolist(),
        val_loss.tolist(),
        train_acc.tolist(),
        val_acc.tolist(),
    )


def build_summary_rows(rng: np.random.Generator) -> list[dict]:
    rows = []
    for variant in SUPPORTED_VARIANTS:
        t = VARIANT_TARGETS[variant]
        acc = t["val_acc"] + rng.uniform(-0.002, 0.002)
        macro_f1 = t["macro_f1"] + rng.uniform(-0.002, 0.002)
        weighted_f1 = macro_f1 + rng.uniform(0.004, 0.012)
        prec = macro_f1 + rng.uniform(0.005, 0.015)
        rec = macro_f1 + rng.uniform(-0.008, 0.006)
        train_sec = rng.uniform(2400, 3200) + (VARIANT_PARAMS[variant] - 88_655_015) / 500_000

        rows.append(
            {
                "Variant": variant,
                "Accuracy": round(acc, 6),
                "Macro_F1": round(macro_f1, 6),
                "Weighted_F1": round(min(weighted_f1, acc + 0.01), 6),
                "Precision": round(min(prec, acc + 0.02), 6),
                "Recall": round(rec, 6),
                "Parameters": VARIANT_PARAMS[variant],
                "Best_Epoch": t["best_epoch"],
                "Training_Time": round(train_sec, 2),
                "Inference_ms": round(INFERENCE_MS[variant] + rng.uniform(-1.5, 1.5), 4),
            }
        )
    return rows


def save_loss_plot_illustrative(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str,
    title: str,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=9, style="italic", color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_acc_plot_illustrative(
    train_accs: list[float],
    val_accs: list[float],
    save_path: str,
    title: str,
) -> None:
    epochs = range(1, len(train_accs) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [a * 100 for a in train_accs], label="Train Acc", linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, [a * 100 for a in val_accs], label="Val Acc", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=9, style="italic", color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def write_summary_csv(rows: list[dict], path: Path) -> None:
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
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_ablation_comparison_illustrative(rows: list[dict], save_path: Path) -> None:
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
    ax.set_title(f"Architectural Ablation — Accuracy vs Macro F1 ({TITLE_SUFFIX})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    y_min = min(acc + f1) - 4
    ax.set_ylim(max(75, y_min), 100)

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

    fig.text(0.5, 0.02, SUBTITLE, ha="center", fontsize=9, style="italic", color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def write_history_csv(
    path: Path,
    train_loss: list[float],
    val_loss: list[float],
    train_acc: list[float],
    val_acc: list[float],
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"])
        for i in range(len(train_loss)):
            lr = 0.001 * (0.5 * (1 + np.cos(np.pi * i / max(N_EPOCHS - 1, 1))))
            writer.writerow([i + 1, train_loss[i], train_acc[i], val_loss[i], val_acc[i], lr])


def plot_confusion_matrix_illustrative(rng: np.random.Generator, save_path: Path, n_classes: int = 38) -> None:
    """Row-normalised confusion matrix with strong diagonal and mild off-diagonal confusion."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    per_class = rng.integers(80, 120, size=n_classes)
    for i in range(n_classes):
        correct = int(per_class[i] * rng.uniform(0.88, 0.96))
        cm[i, i] = correct
        remainder = per_class[i] - correct
        if remainder > 0:
            # Confuse with 1–2 neighbouring indices (similar-looking classes)
            neighbors = [(i + 1) % n_classes, (i - 1) % n_classes]
            rng.shuffle(neighbors)
            split = rng.integers(1, max(2, remainder))
            cm[i, neighbors[0]] += split
            cm[i, neighbors[1]] += remainder - split

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = [f"C{i+1:02d}" for i in range(n_classes)]

    fig_size = max(10, n_classes // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Row-normalised count")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted class (illustrative labels)")
    ax.set_ylabel("True class (illustrative labels)")
    ax.set_title(f"Confusion Matrix — multiscale_attention ({TITLE_SUFFIX})")
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=9, style="italic", color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(ILLUSTRATIVE_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_summary_rows(rng)
    write_summary_csv(rows, OUTPUT_DIR / "ablation_summary_ILLUSTRATIVE.csv")
    plot_ablation_comparison_illustrative(rows, OUTPUT_DIR / "ablation_comparison_ILLUSTRATIVE.png")

    for variant in SUPPORTED_VARIANTS:
        variant_dir = OUTPUT_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        t = VARIANT_TARGETS[variant]
        train_loss, val_loss, train_acc, val_acc = synthesize_curves(
            rng,
            final_val_acc=t["val_acc"],
        )
        write_history_csv(
            variant_dir / "history_ILLUSTRATIVE.csv",
            train_loss,
            val_loss,
            train_acc,
            val_acc,
        )
        save_loss_plot_illustrative(
            train_loss,
            val_loss,
            save_path=str(variant_dir / "loss_curve_ILLUSTRATIVE.png"),
            title=f"ConvGeM-NeXt [{variant}] — Train vs Val Loss ({TITLE_SUFFIX})",
        )
        save_acc_plot_illustrative(
            train_acc,
            val_acc,
            save_path=str(variant_dir / "accuracy_curve_ILLUSTRATIVE.png"),
            title=f"ConvGeM-NeXt [{variant}] — Train vs Val Accuracy ({TITLE_SUFFIX})",
        )

    plot_confusion_matrix_illustrative(
        rng,
        OUTPUT_DIR / "multiscale_attention" / "confusion_matrix_ILLUSTRATIVE.png",
    )

    print(f"[teaching] Wrote illustrative materials under → {OUTPUT_DIR}")
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUTPUT_DIR.parent)}")


if __name__ == "__main__":
    main()
