"""
generate_plantdoc_teaching_figures.py — PlantDoc teaching figures from published values.

Every number comes from Arshad et al. (2026), "ConvGeM-next: a deep learning framework
for plant disease detection", Front. Plant Sci. 17:1763739 (table/figure cited per item).
No training is run and no values are invented. The only approximations are the learning
curves, which are traced by eye from the paper's Figure 5; their titles say so.

Output mirrors the results/ layout written by ablation_study.py:

  teaching_material_PlantDoc/
    ablation_summary.csv        same columns as results/ablation_summary.csv (+ Source)
    ablation_comparison.png
    model_comparison.{csv,png}  Table 11 (all models compared in the paper)
    convnext_base/metrics.json  plain ConvNeXt-Base baseline (§4.3)
    baseline/                   ConvGeM-NeXt = this repo's "baseline" variant
      metrics.json, history.csv, loss_curve.png, classification_report.csv,
      classwise_f1.png, confusion_matrix.csv, confusion_matrix.png

The paper reports no multiscale / attention variants, so those folders are not
created. Columns the paper does not report are left blank.

Usage:
    python generate_plantdoc_teaching_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "teaching_material_PlantDoc"
SOURCE = "Source: Arshad et al., Front. Plant Sci. 2026 (doi:10.3389/fpls.2026.1763739)"

# ---- Table 8: class-wise PlantDoc results (precision, recall, F1 in %) -------
TABLE8 = [
    ("Apple_Scab_Leaf", 100, 100, 100), ("Apple_leaf", 50, 100, 66.67),
    ("Apple_rust_leaf", 100, 100, 100), ("Bell_pepper_leaf", 100, 100, 100),
    ("Bell_pepper_leaf_spot", 100, 100, 100), ("Blueberry_leaf", 100, 100, 100),
    ("Cherry_leaf", 100, 100, 100), ("Corn_Gray_leaf_spot", 100, 100, 100),
    ("Corn_leaf_blight", 100, 100, 100), ("Corn_rust_leaf", 100, 100, 100),
    ("Peach_leaf", 100, 100, 100), ("Potato_leaf_early_blight", 50, 50, 50),
    ("Potato_leaf_late_blight", 0, 0, 0), ("Raspberry_leaf", 100, 100, 100),
    ("Soyabean_leaf", 100, 100, 100), ("Squash_Powdery_mildew_leaf", 100, 100, 100),
    ("Strawberry_leaf", 100, 100, 100), ("Tomato_Early_blight_leaf", 50, 100, 66.67),
    ("Tomato_Septoria_leaf_spot", 100, 100, 100), ("Tomato_leaf", 100, 100, 100),
    ("Tomato_leaf_bacterial_spot", 100, 100, 100), ("Tomato_leaf_late_blight", 100, 100, 100),
    ("Tomato_leaf_mosaic_virus", 0, 0, 0), ("Tomato_leaf_yellow_virus", 0, 0, 0),
    ("Tomato_mold_leaf", 33.33, 100, 50), ("grape_leaf", 100, 100, 100),
    ("grape_leaf_black_rot", 100, 100, 100),
]
CLASSES = [r[0] for r in TABLE8]

# ---- Figure 7: off-diagonal mass of the row-normalised confusion matrix ------
# (true class → {predicted class: %}); all other rows are 100 % on the diagonal.
FIG7_OFF_DIAG = {
    "Potato_leaf_early_blight": {"Potato_leaf_early_blight": 50, "Tomato_Early_blight_leaf": 50},
    "Potato_leaf_late_blight": {"Potato_leaf_early_blight": 100},
    "Tomato_leaf_mosaic_virus": {"Tomato_mold_leaf": 100},
    "Tomato_leaf_yellow_virus": {"Apple_leaf": 50, "Tomato_mold_leaf": 50},
}

# ---- Tables 6, 7 and §4.3 ----------------------------------------------------
SUMMARY = [
    # Model, Accuracy %, Macro precision, Macro F1, MCA %, Test loss, Source
    ("ConvNeXt-Base (baseline)", 91.84, "", 0.9231, "", "", "§4.3"),
    ("ConvGeM-NeXt (best run)", 94.69, 0.9477, 0.9465, 87.04, 0.2552, "Table 6, §4.4.1"),
    ("ConvGeM-NeXt (3 seeds, mean ± std)", "92.19 ± 2.58", "0.923 ± 0.025", "0.923 ± 0.025", "", "", "Table 7"),
]

# ---- Table 11: PlantDoc accuracy of compared models --------------------------
TABLE11 = [
    ("MobileNet-V2", 47.56, None), ("DenseNet-121", 56.5, "55–58"),
    ("EfficientNet-B0", 59.5, "57–62"), ("ResNet-50 + CBAM", 69.52, None),
    ("ViT-RoT", 72.85, None), ("Swin Transformer", 77.54, None),
    ("ConvGeM-NeXt", 94.69, None),
]

# ---- Figure 5: approximate trace (epoch, train loss, val loss) ---------------
FIG5 = [(0, 1.75, 1.03), (3, 0.60, 0.72), (6, 0.28, 0.50), (9, 0.13, 0.23), (12, 0.07, 0.22),
        (15, 0.05, 0.25), (18, 0.03, 0.19), (21, 0.02, 0.19), (24, 0.015, 0.19), (29, 0.01, 0.19)]


SRC_SHORT = "Arshad et al. 2026, Front. Plant Sci. 17:1763739"
PARAMS_CONVGEM = 88_655_015  # counted from model.py (paper: "about 89 million", §3.2.3)
SUMMARY_FIELDS = ["Variant", "Accuracy", "Macro_F1", "Weighted_F1", "Precision", "Recall",
                  "Parameters", "Best_Epoch", "Training_Time", "Inference_ms", "Source"]


def footer(fig) -> None:
    fig.text(0.5, 0.005, SOURCE, ha="center", va="bottom", fontsize=8, color="#555")


def save(fig, path: Path) -> None:
    footer(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(path: Path, header: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def summary() -> None:
    rows = [
        ["convnext_base", 0.9184, 0.9231, "", "", "", "", "", "", "",
         f"{SRC_SHORT}, §4.3 (Accuracy, Macro_F1)"],
        ["baseline (ConvGeM-NeXt)", 0.9469, 0.9465, "", 0.9477, "", PARAMS_CONVGEM, "", "", "",
         f"{SRC_SHORT}, Table 6 / §4.3; Parameters counted from model.py"],
    ]
    write_csv(OUT / "ablation_summary.csv", SUMMARY_FIELDS, rows)

    names = ["ConvNeXt-Base", "ConvGeM-NeXt"]
    acc, f1 = [91.84, 94.69], [92.31, 94.65]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for off, vals, lab in ((-0.18, acc, "Accuracy (%)"), (0.18, f1, "Macro F1 (%)")):
        bars = ax.bar(x + off, vals, 0.36, label=lab)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(80, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("PlantDoc — Accuracy vs Macro F1 (paper §4.3, Table 6)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, OUT / "ablation_comparison.png")


def model_comparison() -> None:
    write_csv(OUT / "model_comparison.csv", ["Model", "PlantDoc_Accuracy_%", "Reported_as", "Source"],
              [[n, a, r or a, f"{SRC_SHORT}, Table 11"] for n, a, r in TABLE11])
    names = [m for m, _, _ in TABLE11]
    accs = [a for _, a, _ in TABLE11]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(names, accs, color=["#9aa5b1"] * (len(names) - 1) + ["#1b7f4b"])
    for b, (_, a, rng) in zip(bars, TABLE11):
        ax.annotate(f"{rng or a}%", (a, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("PlantDoc test accuracy (%)")
    ax.set_title("PlantDoc accuracy of compared models (paper Table 11)")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, OUT / "model_comparison.png")


def convnext_base() -> None:
    d = OUT / "convnext_base"
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "metrics.json", {
        "variant": "convnext_base", "dataset_name": "PlantDoc", "num_classes": 27,
        "accuracy": 0.9184, "macro_f1": 0.9231,
        "source": f"{SRC_SHORT}, §4.3",
        "note": "Plain ConvNeXt-Base with GAP and the standard head. The paper reports only these two values.",
    })


def convgem() -> None:
    d = OUT / "baseline"
    d.mkdir(parents=True, exist_ok=True)
    recomputed_f1 = float(np.mean([r[3] for r in TABLE8])) / 100
    write_json(d / "metrics.json", {
        "variant": "baseline", "model": "ConvGeM-NeXt", "dataset_name": "PlantDoc", "num_classes": 27,
        "accuracy": 0.9469, "precision": 0.9477, "macro_f1": 0.9465, "mca": 0.8704, "test_loss": 0.2552,
        "three_seed_accuracy_mean": 0.9219, "three_seed_accuracy_std": 0.0258,
        "three_seed_macro_f1_mean": 0.923, "three_seed_macro_f1_std": 0.025,
        "parameters": PARAMS_CONVGEM,
        "macro_f1_recomputed_from_table8": round(recomputed_f1, 4),
        "source": f"{SRC_SHORT}: Table 6, Table 7, §4.3, §4.4.1, Table 8; parameters counted from model.py",
    })

    # classification_report.csv — same columns as evaluate.py (+ source); support not reported
    rows = [[c, p / 100, r / 100, f / 100, "", f"{SRC_SHORT}, Table 8"] for c, p, r, f in TABLE8]
    rows.append(["macro avg", 0.9477, "", 0.9465, "", f"{SRC_SHORT}, Table 6 (reported)"])
    rows.append(["macro avg (recomputed from Table 8)",
                 round(float(np.mean([r[1] for r in TABLE8])) / 100, 4),
                 round(float(np.mean([r[2] for r in TABLE8])) / 100, 4),
                 round(recomputed_f1, 4), "", "Arithmetic mean of Table 8 rows"])
    write_csv(d / "classification_report.csv",
              ["class", "precision", "recall", "f1-score", "support", "source"], rows)

    # history.csv — loss only (Fig. 5 shows loss; accuracy curve not published)
    pts = np.array(FIG5)
    ep = np.arange(0, 30)
    train = np.interp(ep, pts[:, 0], pts[:, 1])
    val = np.interp(ep, pts[:, 0], pts[:, 2])
    write_csv(d / "history.csv",
              ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr", "source"],
              [[int(e) + 1, f"{t:.4f}", "", f"{v:.4f}", "",
                f"{1e-3 * 0.5 * (1 + np.cos(np.pi * e / 30)):.6e}",
                "Approximate trace of Fig. 5 (loss); lr from Table 2 schedule"]
               for e, t, v in zip(ep, train, val)])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep + 1, train, lw=2, marker="o", ms=3, label="Train Loss")
    ax.plot(ep + 1, val, lw=2, marker="s", ms=3, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("ConvGeM-NeXt [baseline] — Train vs Val Loss, PlantDoc\n(approximate trace of paper Fig. 5)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    save(fig, d / "loss_curve.png")

    # per-class F1
    order = np.argsort([r[3] for r in TABLE8])
    f1 = [TABLE8[i][3] for i in order]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh([CLASSES[i] for i in order], f1,
            color=["#c0392b" if v == 0 else "#e67e22" if v < 100 else "#1b7f4b" for v in f1])
    ax.set_xlim(0, 105)
    ax.set_xlabel("F1-score (%)")
    ax.set_title("ConvGeM-NeXt per-class F1 on PlantDoc (paper Table 8)")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, d / "classwise_f1.png")

    # confusion matrix (row-normalised %, Fig. 7)
    n = len(CLASSES)
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.eye(n) * 100
    for true, preds in FIG7_OFF_DIAG.items():
        cm[idx[true]] = 0
        for pred, v in preds.items():
            cm[idx[true], idx[pred]] = v
    with open(d / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# row-normalised % (not counts); reconstructed from " + SRC_SHORT + ", Fig. 7 / Table 8"])
        w.writerow(["true\\pred"] + CLASSES)
        for c, row in zip(CLASSES, cm):
            w.writerow([c] + [f"{v:.0f}" for v in row])
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 60 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
    ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — PlantDoc [baseline] (row-normalised %, reconstructed from paper Fig. 7)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    save(fig, d / "confusion_matrix.png")


def main() -> None:
    import shutil
    if OUT.exists():
        for p in OUT.iterdir():
            if p.name != "README.md":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    summary()
    model_comparison()
    convnext_base()
    convgem()
    print(f"PlantDoc teaching results written to {OUT}")


if __name__ == "__main__":
    main()
