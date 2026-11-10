"""
make_paper_figures.py — rebuild every figure in the paper from the CSV/JSON
files in ../results.  Nothing is typed in by hand except the base-paper
reference values (Arshad et al., 2026), which are labelled as such.

PlantDoc ablation and cross-corpus inputs carry an ILLUSTRATIVE note in the
source files; every figure built from them is hatched and stamped with "*".

Usage:  python paper/make_paper_figures.py
Output: paper/figures/*.png  and  paper/figures/numbers.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PV = RES / "Plant VIllage Result"
PD = RES / "Plant Doc Result"
CC = RES / "cross_corpus"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["baseline", "multiscale", "attention", "multiscale_attention"]
LABELS = {
    "baseline": "Baseline\n(ConvGeM-NeXt)",
    "multiscale": "+ MSFF",
    "attention": "+ CBAM",
    "multiscale_attention": "+ MSFF + CBAM\n(proposed)",
}
SHORT = {
    "baseline": "Baseline",
    "multiscale": "+MSFF",
    "attention": "+CBAM",
    "multiscale_attention": "Proposed",
}
COLORS = {
    "baseline": "#7f8c9a",
    "multiscale": "#2f7fc1",
    "attention": "#8e5bb5",
    "multiscale_attention": "#d9822b",
}
PAPER_BLUE = "#1f5fa8"
PAPER_RED = "#b03a2e"

# Base-paper reference values (Arshad et al., 2026, Frontiers in Plant Science)
PAPER = {
    "pv_acc": 99.65, "pv_f1": 99.59, "pv_baseline_convnext": 96.19, "pv_gem_only": 96.97,
    "pd_acc": 94.69, "pd_f1": 94.65, "pd_prec": 94.77, "pd_mca": 87.04,
    "pv_to_pd": 41.00, "pd_to_pv": 53.23,
    "pv_to_pd_f1": 17.29, "pd_to_pv_f1": 46.85,
}

ILLUS = "* Illustrative placeholder — not a measured result"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stamp(fig, text=ILLUS):
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7.5,
             color=PAPER_RED, style="italic")


def bar_labels(ax, bars, fmt="{:.2f}", dy=0.25, size=7.5):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=size)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
pv_sum = {r["Variant"]: r for r in read_csv(PV / "ablation_summary.csv")}
pd_sum = {r["Variant"]: r for r in read_csv(PD / "ablation_summary.csv")}
pv_hist = {v: read_csv(PV / v / "history.csv") for v in VARIANTS}
pd_hist = {v: read_csv(PD / v / "history.csv") for v in VARIANTS}
cc_rows = read_csv(CC / "cross_corpus_summary.csv")


def load_cm(path: Path, has_row_labels: bool):
    rows = list(csv.reader(open(path)))
    header = rows[0][1:] if has_row_labels else rows[0]
    body = [r[1:] if has_row_labels else r for r in rows[1:]]
    return header, np.array([[float(x) for x in r] for r in body])


pv_cls, pv_cm = load_cm(PV / "multiscale_attention" / "confusion_matrix.csv", False)
pd_cls, pd_cm = load_cm(PD / "multiscale_attention" / "confusion_matrix.csv", True)


def pretty_pv(name: str) -> str:
    n = name.replace("Pepper__bell___", "Pepper ").replace("Potato___", "Potato ")
    n = n.replace("Tomato__Tomato_", "Tomato ").replace("Tomato__", "Tomato ").replace("Tomato_", "Tomato ")
    n = n.replace("Spider_mites_Two_spotted_spider_mite", "Spider mites")
    n = n.replace("YellowLeaf__Curl_Virus", "YLC virus").replace("_", " ")
    return " ".join(n.split())


def per_class(cm):
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


# ---------------------------------------------------------------------------
# Figure 1 — architecture of the proposed model
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 36)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, ec="#33475b", fs=8, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                           fc=fc, ec=ec, lw=1.0)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                weight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#33475b", lw=1.1))

    # group frames
    for (x, w, title, col) in [(0.5, 13, "Input", "#eef3f8"),
                              (15, 37, "ConvNeXt-Base backbone", "#e9f5ec"),
                              (54, 26, "Proposed neck", "#fdf0e3"),
                              (82, 22.5, "Classification head", "#eef0fb")]:
        ax.add_patch(FancyBboxPatch((x, 1), w, 33, boxstyle="round,pad=0.2,rounding_size=1.5",
                                    fc=col, ec="#9aa7b4", lw=0.8, ls="--"))
        ax.text(x + w / 2, 32.3, title, ha="center", va="center", fontsize=8.2, weight="bold",
                color="#33475b")

    box(2, 19, 10, 8, "RGB leaf\nimage", "white")
    box(2, 6, 10, 9, "Resize 224×224\nImageNet\nnormalise", "white", fs=7.2)
    arrow(7, 19, 7, 15.4)

    stages = [("Stage 1\n56×56\n×128", 17), ("Stage 2\n28×28\n×256", 26),
              ("Stage 3\n14×14\n×512", 35), ("Stage 4\n7×7\n×1024", 44)]
    for i, (t, x) in enumerate(stages):
        box(x, 13, 7.2, 10, t, "#cfe8d5", fs=7)
        if i:
            arrow(stages[i - 1][1] + 7.4, 18, x - 0.2, 18)
    arrow(12.2, 10.5, 16.8, 16)

    box(56, 21, 22, 7, "Multi-Scale Feature Fusion\n1×1 proj → 256 ch, upsample,\nconcat → 3×3 conv",
        "#f9d9b8", fs=6.8)
    box(56, 12, 22, 6.5, "CBAM attention\nchannel (GAP-MLP) → spatial (7×7)", "#f9d9b8", fs=6.8)
    box(56, 3.5, 22, 6, "GeM pooling  (p learnable, init 3)\n28×28×256 → 256", "#f6c79a", fs=6.8)
    # skip arrows from stages 2-4 into MSFF
    for x in (29.6, 38.6, 47.6):
        ax.annotate("", xy=(56, 25), xytext=(x, 23.2),
                    arrowprops=dict(arrowstyle="-|>", color="#c0712b", lw=0.9,
                                    connectionstyle="arc3,rad=-0.25"))
    arrow(67, 21, 67, 18.7)
    arrow(67, 12, 67, 9.7)

    head = ["Linear 256→1024", "BatchNorm1d", "ReLU", "Dropout 0.3", "Linear 1024→C"]
    for i, t in enumerate(head):
        box(84, 25 - i * 5, 18.5, 3.6, t, "#d6dcf5", fs=7)
        if i:
            arrow(93.25, 25 - (i - 1) * 5, 93.25, 25 - i * 5 + 3.8)
    ax.annotate("", xy=(83.8, 26.8), xytext=(78.2, 6.5),
                arrowprops=dict(arrowstyle="-|>", color="#33475b", lw=1.1,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(93.25, 2.6, "Disease class (softmax)", ha="center", fontsize=7.5, style="italic")
    arrow(93.25, 5, 93.25, 3.6)
    fig.savefig(OUT / "fig01_architecture.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ablation bar chart (PV measured / PD illustrative)
# ---------------------------------------------------------------------------
def fig_ablation(summary, fname, title, illustrative, ylim, paper_line=None):
    metrics = [("Accuracy", "Accuracy"), ("Macro_F1", "Macro F1"),
               ("Precision", "Macro precision"), ("Recall", "Macro recall")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(metrics))
    w = 0.2
    for i, v in enumerate(VARIANTS):
        vals = [float(summary[v][k]) * 100 for k, _ in metrics]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=COLORS[v], label=SHORT[v],
                      hatch="///" if illustrative else None, edgecolor="white", lw=0.4)
        bar_labels(ax, bars, "{:.1f}", dy=0.15, size=6.3)
    if paper_line is not None:
        ax.axhline(paper_line[0], color=PAPER_RED, ls="--", lw=1)
        ax.text(3.55, paper_line[0] + 0.15, paper_line[1], color=PAPER_RED, fontsize=7,
                ha="right", va="bottom")
    ax.set_xticks(x, [m for _, m in metrics])
    ax.set_ylim(*ylim)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    if illustrative:
        stamp(fig)
    fig.savefig(OUT / fname)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training curves (2×2)
# ---------------------------------------------------------------------------
def fig_curves(hist, fname, ds, illustrative):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2))
    h = hist["multiscale_attention"]
    ep = [int(r["epoch"]) for r in h]
    f = lambda rows, k: [float(r[k]) for r in rows]
    axs[0, 0].plot(ep, f(h, "train_loss"), "-o", ms=2.5, color="#2f7fc1", label="Train")
    axs[0, 0].plot(ep, f(h, "val_loss"), "-s", ms=2.5, color="#d9822b", label="Validation")
    axs[0, 0].set_title("(a) Proposed — loss")
    axs[0, 0].set_ylabel("Cross-entropy loss")
    axs[0, 1].plot(ep, [x * 100 for x in f(h, "train_accuracy")], "-o", ms=2.5, color="#2f7fc1", label="Train")
    axs[0, 1].plot(ep, [x * 100 for x in f(h, "val_accuracy")], "-s", ms=2.5, color="#d9822b", label="Validation")
    axs[0, 1].set_title("(b) Proposed — accuracy")
    axs[0, 1].set_ylabel("Accuracy (%)")
    for v in VARIANTS:
        e = [int(r["epoch"]) for r in hist[v]]
        axs[1, 0].plot(e, [x * 100 for x in f(hist[v], "val_accuracy")], color=COLORS[v], lw=1.6, label=SHORT[v])
        axs[1, 1].plot(e, f(hist[v], "val_loss"), color=COLORS[v], lw=1.6, label=SHORT[v])
    axs[1, 0].set_title("(c) Validation accuracy — all variants")
    axs[1, 0].set_ylabel("Accuracy (%)")
    axs[1, 1].set_title("(d) Validation loss — all variants")
    axs[1, 1].set_ylabel("Cross-entropy loss")
    for a in axs.flat:
        a.set_xlabel("Epoch")
        a.legend(fontsize=7)
    fig.suptitle(ds, fontsize=10, weight="bold", y=1.0)
    fig.tight_layout()
    if illustrative:
        stamp(fig)
    fig.savefig(OUT / fname)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion matrix (row-normalised %)
# ---------------------------------------------------------------------------
def fig_cm(cm, names, fname, title, illustrative, cmap, size):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=size)
    ax.grid(False)
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=100)
    n = len(names)
    ax.set_xticks(range(n), names, rotation=90, fontsize=6.3)
    ax.set_yticks(range(n), names, fontsize=6.3)
    for i in range(n):
        for j in range(n):
            val = norm[i, j]
            if val >= 0.5:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5.2,
                        color="white" if val > 55 else "#222")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Row-normalised (%)", fontsize=7)
    if illustrative:
        stamp(fig)
    fig.savefig(OUT / fname)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PV per-class P/R/F1 (derived from the measured confusion matrix)
# ---------------------------------------------------------------------------
def fig_pv_perclass():
    p, r, f1 = per_class(pv_cm)
    names = [pretty_pv(c) for c in pv_cls]
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(names))
    ax.barh(y - 0.25, p[order] * 100, 0.25, color="#2f7fc1", label="Precision")
    ax.barh(y, r[order] * 100, 0.25, color="#d9822b", label="Recall")
    ax.barh(y + 0.25, f1[order] * 100, 0.25, color="#3d9a5b", label="F1-score")
    ax.set_yticks(y, [names[i] for i in order], fontsize=7)
    ax.set_xlim(70, 101)
    ax.set_xlabel("Score (%)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Per-class precision / recall / F1 — proposed model, PlantVillage (15 classes)")
    fig.savefig(OUT / "fig05_pv_perclass.png")
    plt.close(fig)
    return names, p, r, f1


# ---------------------------------------------------------------------------
# Cost vs accuracy (PV measured)
# ---------------------------------------------------------------------------
def fig_cost():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for v in VARIANTS:
        s = pv_sum[v]
        params = float(s["Parameters"]) / 1e6
        acc = float(s["Accuracy"]) * 100
        ms = float(s["Inference_ms"])
        axs[0].scatter(params, acc, s=70, color=COLORS[v], zorder=3, label=SHORT[v])
        axs[1].scatter(ms, acc, s=70, color=COLORS[v], zorder=3, label=SHORT[v])
    axs[0].set_xlabel("Parameters (M)")
    axs[1].set_xlabel("Inference time per batch (ms)")
    for a in axs:
        a.set_ylabel("Test accuracy (%)")
    axs[0].set_title("(a) Accuracy vs. model size")
    axs[1].set_title("(b) Accuracy vs. inference cost")
    axs[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig06_cost.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-corpus vs paper Table 14
# ---------------------------------------------------------------------------
def fig_cross_corpus():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (src, tgt, paper_acc) in zip(axs, [("PlantVillage", "PlantDoc", PAPER["pv_to_pd"]),
                                              ("PlantDoc", "PlantVillage", PAPER["pd_to_pv"])]):
        rows = {r["Variant"]: r for r in cc_rows if r["Source"] == src and r["Target"] == tgt}
        names = ["Base paper\n(Table 14)"] + [SHORT[v] for v in VARIANTS]
        vals = [paper_acc] + [float(rows[v]["Accuracy"]) * 100 for v in VARIANTS]
        cols = [PAPER_BLUE] + [COLORS[v] for v in VARIANTS]
        hatches = [None] + ["///"] * 4
        bars = ax.bar(range(5), vals, color=cols, edgecolor="white", lw=0.4)
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
        bar_labels(ax, bars, "{:.1f}", dy=0.6, size=7)
        ax.set_xticks(range(5), names, fontsize=7)
        ax.set_title(f"{src} → {tgt} (no fine-tuning)")
        ax.set_ylim(0, 70)
    axs[0].set_ylabel("Top-1 accuracy (%)")
    fig.tight_layout()
    stamp(fig, "* Hatched bars: illustrative placeholders — not measured. Solid blue: base-paper values.")
    fig.savefig(OUT / "fig10_cross_corpus.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison with base paper
# ---------------------------------------------------------------------------
def fig_vs_paper():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) relative gain of each design over its own baseline
    ax = axs[0]
    labels = ["Paper\nGAP→GeM", "Paper\nfull model", "Ours\n+MSFF", "Ours\n+CBAM", "Ours\nproposed"]
    base_pv = float(pv_sum["baseline"]["Accuracy"]) * 100
    gains = [PAPER["pv_gem_only"] - PAPER["pv_baseline_convnext"],
             PAPER["pv_acc"] - PAPER["pv_baseline_convnext"],
             float(pv_sum["multiscale"]["Accuracy"]) * 100 - base_pv,
             float(pv_sum["attention"]["Accuracy"]) * 100 - base_pv,
             float(pv_sum["multiscale_attention"]["Accuracy"]) * 100 - base_pv]
    cols = [PAPER_BLUE, PAPER_BLUE, COLORS["multiscale"], COLORS["attention"], COLORS["multiscale_attention"]]
    bars = ax.bar(range(5), gains, color=cols)
    bar_labels(ax, bars, "+{:.2f}", dy=0.05, size=7)
    ax.set_xticks(range(5), labels, fontsize=7)
    ax.set_ylabel("Gain over own baseline (pp)")
    ax.set_title("(a) PlantVillage: gain over baseline")
    ax.set_ylim(0, 7)

    # (b) PlantDoc absolute: paper vs ours*
    ax = axs[1]
    met = ["Accuracy", "Macro F1", "Macro precision"]
    paper_vals = [PAPER["pd_acc"], PAPER["pd_f1"], PAPER["pd_prec"]]
    ours = pd_sum["multiscale_attention"]
    our_vals = [float(ours["Accuracy"]) * 100, float(ours["Macro_F1"]) * 100, float(ours["Precision"]) * 100]
    x = np.arange(3)
    b1 = ax.bar(x - 0.18, paper_vals, 0.36, color=PAPER_BLUE, label="Base paper (reported)")
    b2 = ax.bar(x + 0.18, our_vals, 0.36, color=COLORS["multiscale_attention"], hatch="///",
                edgecolor="white", lw=0.4, label="Proposed*")
    bar_labels(ax, b1, "{:.2f}", dy=0.1, size=6.5)
    bar_labels(ax, b2, "{:.2f}", dy=0.1, size=6.5)
    ax.set_xticks(x, met, fontsize=7.5)
    ax.set_ylim(90, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("(b) PlantDoc: base paper vs. proposed*")
    ax.legend(fontsize=7, loc="upper left", ncol=1, bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    stamp(fig, "* PlantDoc values for the proposed model are illustrative placeholders — not measured.")
    fig.savefig(OUT / "fig11_vs_base_paper.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison with literature (paper Table 11/12 values + ours)
# ---------------------------------------------------------------------------
def fig_literature():
    lit = [("MobileNet-V2", 47.56), ("DenseNet-121", 56.5), ("EfficientNet-B0", 59.5),
           ("ResNet-50+CBAM", 69.52), ("RF-HOG-SVM", 72.4), ("ViT-RoT", 72.85),
           ("Swin Transformer", 77.54), ("DeepLeaf", 85.6), ("LeafDoc++", 88.9),
           ("AgriVision", 90.1), ("ConvGeM-NeXt\n(base paper)", PAPER["pd_acc"])]
    ours = float(pd_sum["multiscale_attention"]["Accuracy"]) * 100
    names = [n for n, _ in lit] + ["Proposed*"]
    vals = [v for _, v in lit] + [ours]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(names))
    cols = ["#b8c4d0"] * (len(lit) - 1) + [PAPER_BLUE, COLORS["multiscale_attention"]]
    bars = ax.barh(y, vals, color=cols, edgecolor="white", lw=0.4)
    bars[-1].set_hatch("///")
    for b, v in zip(bars, vals):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(y, names, fontsize=7.5)
    ax.set_xlim(40, 102)
    ax.set_xlabel("PlantDoc Top-1 accuracy (%)")
    ax.set_title("PlantDoc accuracy: literature (as reported in base paper Tables 11–12) vs. proposed*")
    stamp(fig, "* Illustrative placeholder — not measured. DenseNet/EfficientNet: mid-point of reported range.")
    fig.savefig(OUT / "fig12_literature.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig_architecture()
    fig_ablation(pv_sum, "fig02_pv_ablation.png",
                 "Architectural ablation — PlantVillage (15 classes, measured)", False, (84, 98.5))
    fig_curves(pv_hist, "fig03_pv_curves.png", "PlantVillage (15 classes) — training dynamics", False)
    fig_cm(pv_cm, [pretty_pv(c) for c in pv_cls], "fig04_pv_confusion.png",
           "Proposed model — PlantVillage (15 classes)", False, "Blues", (6.4, 5.6))
    names, p, r, f1 = fig_pv_perclass()
    fig_cost()
    fig_ablation(pd_sum, "fig07_pd_ablation.png",
                 "Architectural ablation — PlantDoc (27 classes)*", True, (84, 99.5),
                 paper_line=(PAPER["pd_acc"], "Base paper accuracy 94.69%"))
    fig_curves(pd_hist, "fig08_pd_curves.png", "PlantDoc (27 classes) — training dynamics*", True)
    fig_cm(pd_cm, pd_cls, "fig09_pd_confusion.png", "Proposed model — PlantDoc (27 classes)*",
           True, "Greens", (7.2, 6.4))
    fig_cross_corpus()
    fig_vs_paper()
    fig_literature()

    # numbers used in the text, all derived from files
    tp = np.diag(pv_cm)
    numbers = {
        "pv_cm_accuracy": float(tp.sum() / pv_cm.sum()),
        "pv_cm_mca": float(np.mean(tp / pv_cm.sum(1))),
        "pv_cm_n": int(pv_cm.sum()),
        "pv_per_class": {n: {"precision": float(a), "recall": float(b), "f1": float(c),
                             "support": int(s)}
                         for n, a, b, c, s in zip(names, p, r, f1, pv_cm.sum(1))},
        "pd_cm_accuracy": float(np.diag(pd_cm).sum() / pd_cm.sum()),
        "pd_cm_n": int(pd_cm.sum()),
    }
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps({k: v for k, v in numbers.items() if k != "pv_per_class"}, indent=2))
    for n, d in numbers["pv_per_class"].items():
        print(f"{n:28s} P={d['precision']*100:6.2f} R={d['recall']*100:6.2f} F1={d['f1']*100:6.2f} n={d['support']}")
