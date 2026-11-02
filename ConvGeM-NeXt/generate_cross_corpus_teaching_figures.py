"""
generate_cross_corpus_teaching_figures.py — Illustrative §4.9 cross-corpus (paper-style only).

Matches the published experiment: train on corpus A, test on corpus B, **no target
fine-tuning**. Two directions (Table 14) × four architecture variants (ablation-style).

Does NOT include finetune/scratch (repo extensions, not in the paper).

Output: teaching_material_cross_corpus/

Usage:
    python generate_cross_corpus_teaching_figures.py
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cross_corpus import PD_TO_PV_CLASS_MAP, SUMMARY_FIELDS
from model import SUPPORTED_VARIANTS

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "teaching_material_cross_corpus"
ILLUSTRATIVE_SEED = 20261102

BANNER = "ILLUSTRATIVE — not measured from a real run"
SUBTITLE = "Paper §4.9 only · PV↔PD direct transfer · no target fine-tuning"

# Published ConvGeM-NeXt (single model) — Table 14
PAPER_TABLE14 = {
    ("PlantVillage", "PlantDoc"): {"accuracy": 0.41, "macro_f1": 0.1729, "macro_precision": 0.2238},
    ("PlantDoc", "PlantVillage"): {"accuracy": 0.5323, "macro_f1": 0.4685, "macro_precision": 0.5623},
}

# Illustrative variant sweep: slightly above paper, ~55–60 % band (teaching demo)
ILLUSTRATIVE_PV_TO_PD = {
    "baseline": 0.552,
    "multiscale": 0.562,
    "attention": 0.576,
    "multiscale_attention": 0.594,
}

ILLUSTRATIVE_PD_TO_PV = {
    "baseline": 0.548,
    "multiscale": 0.558,
    "attention": 0.571,
    "multiscale_attention": 0.589,
}

N_TEST = 236
MAPPED_CLASSES = len(PD_TO_PV_CLASS_MAP)


def banner(fig) -> None:
    fig.text(0.5, 0.995, BANNER, ha="center", va="top", fontsize=10,
             color="#b00020", fontweight="bold")
    fig.text(0.5, 0.005, SUBTITLE, ha="center", va="bottom", fontsize=8, color="#555")


def save_fig(fig, path: Path, rect=(0, 0.03, 1, 0.94)) -> None:
    banner(fig)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metrics_from_acc(rng: np.random.Generator, acc: float) -> dict[str, float]:
    macro_f1 = float(np.clip(acc - rng.uniform(0.04, 0.08), 0.05, 0.99))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": float(np.clip(acc - rng.uniform(0.01, 0.03), 0.05, 0.99)),
        "precision": float(np.clip(macro_f1 + rng.uniform(0.0, 0.02), 0.05, 0.99)),
        "recall": float(np.clip(macro_f1 + rng.uniform(-0.02, 0.01), 0.05, 0.99)),
        "mca": float(np.clip(macro_f1 + rng.uniform(-0.01, 0.02), 0.05, 0.99)),
    }


def build_rows(
    rng: np.random.Generator,
    source: str,
    target: str,
    accs: dict[str, float],
) -> list[dict]:
    rows = []
    for variant in SUPPORTED_VARIANTS:
        acc = accs[variant]
        m = metrics_from_acc(rng, acc)
        rows.append({
            "Variant": variant,
            "Protocol": "direct",
            "Source": source,
            "Target": target,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Macro_F1": f"{m['macro_f1']:.4f}",
            "Weighted_F1": f"{m['weighted_f1']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "MCA": f"{m['mca']:.4f}",
            "Mapped_Classes": MAPPED_CLASSES,
            "N_Test": N_TEST,
        })
    return rows


def plot_main_comparison(path: Path) -> None:
    """Two panels: PV→PD and PD→PV; four variant bars each; paper Table 14 as dashed line."""
    variants = list(SUPPORTED_VARIANTS)
    x = np.arange(len(variants))
    width = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)

    configs = [
        (
            axes[0],
            "PlantVillage → PlantDoc\n(train PV · test PD · no fine-tune)",
            ILLUSTRATIVE_PV_TO_PD,
            PAPER_TABLE14[("PlantVillage", "PlantDoc")]["accuracy"] * 100,
            41.0,
        ),
        (
            axes[1],
            "PlantDoc → PlantVillage\n(train PD · test PV · no fine-tune)",
            ILLUSTRATIVE_PD_TO_PV,
            PAPER_TABLE14[("PlantDoc", "PlantVillage")]["accuracy"] * 100,
            53.23,
        ),
    ]

    colors = ["#4daf4a", "#377eb8", "#984ea3", "#ff7f00"]
    for ax, title, accs, paper_acc, paper_label in configs:
        vals = [accs[v] * 100 for v in variants]
        bars = ax.bar(x, vals, width, color=colors)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axhline(paper_acc, color="#b00020", ls="--", lw=1.5,
                   label=f"Paper Table 14 ({paper_label:.2f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20, ha="right")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(0, 68)
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Cross-corpus evaluation by variant (ILLUSTRATIVE · paper §4.9 protocol)",
        fontsize=13,
        y=1.02,
    )
    save_fig(fig, path, rect=(0, 0.04, 1, 0.90))


def write_variant_metrics(rng: np.random.Generator, pair_dir: Path, source: str, target: str,
                          accs: dict[str, float]) -> None:
    for variant in SUPPORTED_VARIANTS:
        out = pair_dir / variant / "direct"
        out.mkdir(parents=True, exist_ok=True)
        acc = accs[variant]
        m = metrics_from_acc(rng, acc)
        payload = {
            "protocol": "direct",
            "variant": variant,
            "source": source,
            "target": target,
            "note": BANNER,
            **m,
            "n_test": N_TEST,
            "mapped_target_classes": MAPPED_CLASSES,
        }
        with open(out / "metrics.json", "w") as f:
            json.dump(payload, f, indent=2)


def write_paper_reference_csv(path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Train", "Test", "Accuracy_pct", "Macro_Precision", "Macro_F1", "Source"])
        w.writerow(["PlantVillage", "PlantDoc", "41.00", "0.2238", "0.1729", "paper Table 14"])
        w.writerow(["PlantDoc", "PlantVillage", "53.23", "0.5623", "0.4685", "paper Table 14"])


def write_readme() -> None:
    pv = ", ".join(f"{v}: {ILLUSTRATIVE_PV_TO_PD[v]*100:.1f}%" for v in SUPPORTED_VARIANTS)
    pd = ", ".join(f"{v}: {ILLUSTRATIVE_PD_TO_PV[v]*100:.1f}%" for v in SUPPORTED_VARIANTS)
    (OUTPUT_DIR / "README.md").write_text(
        "# Cross-corpus teaching material (ILLUSTRATIVE)\n\n"
        f"**{BANNER}**\n\n"
        "Replicates **only** the paper §4.9 / Table 14 protocol:\n\n"
        "- Train on **PlantVillage**, test on **PlantDoc** (no target fine-tuning)\n"
        "- Train on **PlantDoc**, test on **PlantVillage** (reverse)\n\n"
        "Four bars per direction = architecture variants (`baseline` → `multiscale` → "
        "`attention` → `multiscale_attention`), same idea as the ablation table.\n\n"
        "**Not included:** `finetune` / `scratch` (those are repo extensions, not in the paper).\n\n"
        "| Reference | PV→PD | PD→PV |\n"
        "|-----------|-------|-------|\n"
        "| Paper Table 14 | 41.00 % | 53.23 % |\n"
        f"| Illustrative variants | {pv} | {pd} |\n\n"
        "Main figure: `cross_corpus_by_variant.png`. Summary: `cross_corpus_summary.csv`.\n\n"
        "```bash\npython generate_cross_corpus_teaching_figures.py\n```\n",
        encoding="utf-8",
    )


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(ILLUSTRATIVE_SEED)

    write_paper_reference_csv(OUTPUT_DIR / "table14_paper_reference.csv")

    all_rows: list[dict] = []
    pairs = [
        ("PlantVillage", "PlantDoc", ILLUSTRATIVE_PV_TO_PD),
        ("PlantDoc", "PlantVillage", ILLUSTRATIVE_PD_TO_PV),
    ]
    for source, target, accs in pairs:
        pair_dir = OUTPUT_DIR / f"{source}_to_{target}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        rows = build_rows(rng, source, target, accs)
        all_rows.extend(rows)
        with open(pair_dir / "cross_corpus_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(rows)
        write_variant_metrics(rng, pair_dir, source, target, accs)

    with open(OUTPUT_DIR / "cross_corpus_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    plot_main_comparison(OUTPUT_DIR / "cross_corpus_by_variant.png")
    write_readme()
    print(f"Illustrative cross-corpus (paper-style) written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
