"""
make_figures.py — Build the paper figures and tables from MEASURED results
==========================================================================
Reads only files written by real runs (run_experiments.sh → runs/) and redraws every
comparison graph used in the demo paper, with the published ConvGeM-NeXt value
(Arshad et al., 2026) as a dashed reference line. Missing experiments are skipped
and reported.

  Figure                          Reads
  ------------------------------  -----------------------------------------------------------
  fig_summary.png                 all of the below (best variant per experiment vs paper)
  fig_E1_plantvillage.png         runs/plantvillage/ablation_summary.csv
  fig_E2_plantdoc.png             runs/plantdoc/ablation_summary.csv
  fig_E3_E4_cross_corpus.png      runs/cross_corpus/<Src>_to_<Tgt>/<variant>/direct/metrics.json
  fig_E5_improvements.png         runs/improvements/PlantDoc/<variant>/improvement_summary.csv
  all_results.csv                 one row per measured number, with the file it came from

Usage:
  python make_figures.py                      # → runs/figures/
  python make_figures.py --results_dir results_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ["baseline", "multiscale", "attention", "multiscale_attention"]
PAPER = {"E1": 99.65, "E2": 94.69, "E3": 41.00, "E4": 53.23}  # Arshad et al. 2026: Table 6, Table 14
C_ACC, C_F1 = "#1f77b4", "#ff7f0e"


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def grouped(rows: list[dict], ref: float, title: str, path: Path, tag: str) -> None:
    names = [r["Variant"] for r in rows]
    acc = [float(r["Accuracy"]) * 100 for r in rows]
    f1 = [float(r["Macro_F1"]) * 100 for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for off, vals, lab, col in ((-0.18, acc, "Accuracy (%)", C_ACC), (0.18, f1, "Macro F1 (%)", C_F1)):
        bars = ax.bar(x + off, vals, 0.36, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
    ax.axhline(ref, ls="--", color="#b00020", lw=1.3, label=f"Paper ConvGeM-NeXt ({ref:.2f}%)")
    lo = max(0.0, min(acc + f1 + [ref]) - 3)
    ax.set_ylim(lo, min(100.5, max(acc + f1 + [ref]) + (100 - lo) * 0.1))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("Score (%)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.text(0.5, 0.005, tag, ha="center", fontsize=8, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  ✓ {path}")


def main(results: Path) -> None:
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    table: list[dict] = []
    best: dict[str, float] = {}

    # E1 / E2 — architectural ablations
    for key, ds, label in (("E1", "plantvillage", "PlantVillage"), ("E2", "plantdoc", "PlantDoc")):
        path = results / ds / "ablation_summary.csv"
        if not path.is_file():
            print(f"  – {key} {label}: skipped ({path} missing; run: bash run_experiments.sh {key.lower()})")
            continue
        rows = read_csv(path)
        grouped(rows, PAPER[key], f"{key} — {label} architectural ablation (measured)",
                out / f"fig_{key}_{ds}.png", f"Measured: {path}")
        for r in rows:
            table.append({"Experiment": f"{key} {label}", "Variant": r["Variant"],
                          "Accuracy": r["Accuracy"], "Macro_F1": r["Macro_F1"], "File": str(path)})
        best[key] = max(float(r["Accuracy"]) * 100 for r in rows)

    # E3 / E4 — cross-corpus (direct transfer, paper §4.9 protocol)
    pairs = (("E3", "PlantVillage_to_PlantDoc", "PlantVillage → PlantDoc"),
             ("E4", "PlantDoc_to_PlantVillage", "PlantDoc → PlantVillage"))
    found = {}
    for key, pair, label in pairs:
        vals = {}
        for v in VARIANTS:
            m = results / "cross_corpus" / pair / v / "direct" / "metrics.json"
            if m.is_file():
                d = json.loads(m.read_text())
                vals[v] = d["accuracy"] * 100
                table.append({"Experiment": f"{key} {label}", "Variant": v, "Accuracy": d["accuracy"],
                              "Macro_F1": d["macro_f1"], "File": str(m)})
        if vals:
            found[key] = (label, vals)
            best[key] = max(vals.values())
        else:
            print(f"  – {key} {label}: skipped (no direct/metrics.json; run: bash run_experiments.sh {key.lower()})")
    if found:
        fig, axs = plt.subplots(1, len(found), figsize=(5.2 * len(found), 4.4), sharey=True, squeeze=False)
        for ax, (key, (label, vals)) in zip(axs[0], found.items()):
            names = [v for v in VARIANTS if v in vals]
            bars = ax.bar(names, [vals[v] for v in names], color=["#4daf4a", "#377eb8", "#984ea3", "#ff7f00"][:len(names)])
            for b, v in zip(bars, [vals[n] for n in names]):
                ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
            ax.axhline(PAPER[key], ls="--", color="#b00020", lw=1.3, label=f"Paper Table 14 ({PAPER[key]:.2f}%)")
            ax.set_title(f"{key} — {label}", fontsize=10)
            ax.set_ylim(0, 100)
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=15, labelsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        axs[0][0].set_ylabel("Test accuracy (%)")
        fig.suptitle("Cross-corpus zero-shot transfer (measured)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / "fig_E3_E4_cross_corpus.png", dpi=160)
        plt.close(fig)
        print(f"  ✓ {out / 'fig_E3_E4_cross_corpus.png'}")

    # E5 — training improvements (S0–S6)
    imp = sorted((results / "improvements" / "PlantDoc").glob("*/improvement_summary.csv"))
    if imp:
        for path in imp:
            rows = read_csv(path)
            x = np.arange(len(rows))
            acc = [float(r["Accuracy"]) * 100 for r in rows]
            fig, ax = plt.subplots(figsize=(10, 4.8))
            bars = ax.bar(x, acc, color="#1b7f4b")
            for b, v in zip(bars, acc):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
            ax.axhline(PAPER["E2"], ls="--", color="#b00020", lw=1.3, label="Paper ConvGeM-NeXt (94.69%)")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{r['Step']}\n{r['Description'].lstrip('+ ')[:20]}" for r in rows], fontsize=7)
            ax.set_ylim(max(0, min(acc + [PAPER['E2']]) - 5), 100)
            ax.set_ylabel("PlantDoc test accuracy (%)")
            ax.set_title(f"E5 — cumulative training improvements, {path.parent.name} (measured)", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"fig_E5_improvements_{path.parent.name}.png", dpi=160)
            plt.close(fig)
            print(f"  ✓ {out / f'fig_E5_improvements_{path.parent.name}.png'}")
            for r in rows:
                table.append({"Experiment": "E5 PlantDoc improvements", "Variant": f"{path.parent.name} {r['Step']}",
                              "Accuracy": r["Accuracy"], "Macro_F1": r["Macro_F1"], "File": str(path)})
    else:
        print("  – E5 improvements: skipped (run: bash run_experiments.sh e5)")

    # Summary — paper vs best measured variant
    if best:
        keys = [k for k in ("E1", "E2", "E3", "E4") if k in best]
        names = {"E1": "PlantVillage", "E2": "PlantDoc", "E3": "PV → PD", "E4": "PD → PV"}
        x = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for off, vals, lab, col in ((-0.18, [PAPER[k] for k in keys], "Paper ConvGeM-NeXt (published)", "#9aa5b1"),
                                    (0.18, [best[k] for k in keys], "Best variant (measured)", "#1b7f4b")):
            bars = ax.bar(x + off, vals, 0.36, label=lab, color=col)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([names[k] for k in keys])
        ax.set_ylim(0, 108)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Summary — published baseline vs best measured variant", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig_summary.png", dpi=160)
        plt.close(fig)
        print(f"  ✓ {out / 'fig_summary.png'}")

    with open(out / "all_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Experiment", "Variant", "Accuracy", "Macro_F1", "File"])
        w.writeheader()
        w.writerows(table)
    print(f"  ✓ {out / 'all_results.csv'}  ({len(table)} measured rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build figures from measured results.")
    ap.add_argument("--results_dir", default="runs")
    main(Path(ap.parse_args().results_dir))
