"""
cross_corpus.py — Cross-corpus (cross-dataset) evaluation for ConvGeM-NeXt
==========================================================================
Measures how well a model trained on one corpus transfers to another,
typically lab-condition **PlantVillage → in-the-field PlantDoc**.

Protocols (``--protocol``)
--------------------------
direct    Train on the source corpus (or load ``--source_checkpoint``) and
          evaluate on the target test split with **no target training**.
          Logits are restricted to the source classes that have a target
          counterpart (see ``PV_PD_CLASS_MAP``), and predictions are mapped
          into the target label space.
finetune  Initialise from the source model (backbone + neck + head; the
          final layer rows are copied for mapped classes) and fine-tune on
          the target train split. Evaluated on the same target test split.
scratch   Reference: ImageNet-initialised model trained on the target only.
all       Run direct → finetune → scratch with one shared source model.

All protocols use the same target test split (cached in
``data/<target>_split_seed42.json``), so their numbers are comparable.

Usage
-----
  # Full PV → PD study for the paper baseline
  python cross_corpus.py --source plantvillage --target plantdoc --variant baseline

  # Reuse an existing PlantVillage checkpoint, direct transfer only
  python cross_corpus.py --source plantvillage --target plantdoc \
      --source_checkpoint results/baseline/best_model.pth --protocol direct

  # All four architecture variants
  python cross_corpus.py --source plantvillage --target plantdoc --variant all

  # Reverse direction
  python cross_corpus.py --source plantdoc --target plantvillage

  # Synthetic pipeline check
  python cross_corpus.py --source sampletest_plantvillage --target sampletest_plantdoc \
      --epochs 1 --finetune_epochs 1 --batch_size 8 --no_pretrained

Outputs
-------
  results/cross_corpus/<source>_to_<target>/<variant>/
      source/                 (source training run, unless checkpoint given)
      direct/metrics.json, classification_report.csv, confusion_matrix.png
      finetune/…, history.csv, best_model.pth
      scratch/…
  results/cross_corpus/<source>_to_<target>/cross_corpus_summary.csv
  results/cross_corpus/<source>_to_<target>/cross_corpus_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config as cfg
from dataset import get_dataloaders
from evaluate import (
    mean_class_accuracy,
    plot_confusion_matrix,
    save_classification_report_csv,
)
from model import SUPPORTED_VARIANTS, ConvGeMNeXt, count_parameters
from train import evaluate as eval_loss_acc
from train import format_seconds, run_training, save_acc_plot, save_loss_plot, set_seed, train_one_epoch


# ---------------------------------------------------------------------------
# Label-space alignment
# ---------------------------------------------------------------------------

# PlantDoc folder name → PlantVillage folder name. Every one of the 27
# PlantDoc classes has a PlantVillage counterpart. PlantVillage classes with
# no PlantDoc counterpart (e.g. Potato___healthy, Orange___Haunglongbing,
# Tomato___Target_Spot) are excluded from the direct-transfer label space.
PD_TO_PV_CLASS_MAP: Dict[str, str] = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}
PV_TO_PD_CLASS_MAP: Dict[str, str] = {v: k for k, v in PD_TO_PV_CLASS_MAP.items()}


def build_class_alignment(source_classes: List[str], target_classes: List[str]) -> Dict[int, int]:
    """Return ``{target_idx: source_idx}`` for every target class with a source match.

    Identical folder names match directly (e.g. PV-style synthetic → PV);
    otherwise the PlantVillage ↔ PlantDoc table is used in either direction.
    """
    src_index = {c: i for i, c in enumerate(source_classes)}
    alignment: Dict[int, int] = {}
    for t_idx, t_name in enumerate(target_classes):
        candidates = [t_name, PD_TO_PV_CLASS_MAP.get(t_name), PV_TO_PD_CLASS_MAP.get(t_name)]
        for cand in candidates:
            if cand is not None and cand in src_index:
                alignment[t_idx] = src_index[cand]
                break
    return alignment


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mca": mean_class_accuracy(y_true, y_pred, max(labels) + 1),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "n_test": int(len(y_true)),
    }


def write_outputs(
    out_dir: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    metrics: dict,
    title: str,
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    save_classification_report_csv(report_dict, class_names, os.path.join(out_dir, "classification_report.csv"))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"), title)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[cross_corpus] {title}: acc={metrics['accuracy']*100:.2f}%  "
          f"macroF1={metrics['macro_f1']:.4f}  → {out_dir}")


@torch.no_grad()
def predict(model: nn.Module, loader, device, logit_index: Optional[torch.Tensor] = None):
    """Collect ``(y_true, y_pred)``. If *logit_index* is given, argmax runs
    only over those output columns and returns the position within it."""
    model.eval()
    ys, ps = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images).float()
        if logit_index is not None:
            logits = logits.index_select(1, logit_index)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def load_source_model(ckpt_path: str, device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ConvGeMNeXt(
        num_classes=int(ckpt["num_classes"]),
        variant=ckpt.get("variant", "baseline"),
        p_init=cfg.GEM_P_INIT,
        dropout=cfg.DROPOUT,
        pretrained=False,   # weights come from the checkpoint
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def init_target_from_source(
    source_model: ConvGeMNeXt,
    num_target_classes: int,
    alignment: Dict[int, int],
    device,
) -> ConvGeMNeXt:
    """New target-sized model that inherits every source weight.

    The final classifier layer is re-sized; rows of mapped target classes are
    copied from their source class so fine-tuning starts from the direct-
    transfer solution instead of a random head.
    """
    model = ConvGeMNeXt(
        num_classes=num_target_classes,
        variant=source_model.variant,
        p_init=cfg.GEM_P_INIT,
        dropout=cfg.DROPOUT,
        pretrained=False,
    ).to(device)
    src_state = source_model.state_dict()
    tgt_state = model.state_dict()
    final = source_model.head[-1]
    final_prefix = None
    for name, mod in source_model.named_modules():
        if mod is final:
            final_prefix = name
    transferable = {
        k: v for k, v in src_state.items()
        if not k.startswith(f"{final_prefix}.") and k in tgt_state and tgt_state[k].shape == v.shape
    }
    model.load_state_dict(transferable, strict=False)

    tgt_final = model.head[-1]
    with torch.no_grad():
        for t_idx, s_idx in alignment.items():
            tgt_final.weight[t_idx].copy_(final.weight[s_idx])
            tgt_final.bias[t_idx].copy_(final.bias[s_idx])
    print(f"[cross_corpus] Target model initialised from source "
          f"({len(transferable)} tensors, {len(alignment)}/{num_target_classes} head rows copied)")
    return model


def train_on_target(
    model: nn.Module,
    loaders: tuple,
    epochs: int,
    lr: float,
    out_dir: str,
    device,
    label: str,
) -> dict:
    """Plain AdamW + cosine loop on the target train/val split (best val loss kept)."""
    train_loader, val_loader, _ = loaders
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ckpt_path = os.path.join(out_dir, "best_model.pth")
    history_path = os.path.join(out_dir, "history.csv")
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_epoch = float("inf"), 0
    t0 = time.time()
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"])

    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl, va = eval_loss_acc(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tl, ta, vl, va, lr_now])
        print(f"  [{label}] epoch {epoch:>3}/{epochs}  train {tl:.4f}/{ta*100:.2f}%  "
              f"val {vl:.4f}/{va*100:.2f}%")
        if vl < best_val_loss:
            best_val_loss, best_epoch = vl, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": vl, "val_accuracy": va,
                        "num_classes": model.num_classes, "variant": model.variant}, ckpt_path)

    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    if hist["train_loss"]:
        save_loss_plot(hist["train_loss"], hist["val_loss"], os.path.join(out_dir, "loss_curve.png"),
                       title=f"Cross-corpus [{label}] — Loss")
        save_acc_plot(hist["train_acc"], hist["val_acc"], os.path.join(out_dir, "accuracy_curve.png"),
                      title=f"Cross-corpus [{label}] — Accuracy")
    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "training_time_sec": time.time() - t0}


# ---------------------------------------------------------------------------
# One variant, all requested protocols
# ---------------------------------------------------------------------------

def run_variant(args, variant: str, pair_dir: str, device) -> List[dict]:
    source = cfg.resolve_dataset_name(args.source)
    target = cfg.resolve_dataset_name(args.target)
    vdir = os.path.join(pair_dir, variant)
    protocols = ["direct", "finetune", "scratch"] if args.protocol == "all" else [args.protocol]
    rows: List[dict] = []

    num_workers = cfg.NUM_WORKERS if device.type == "cuda" else 0
    pin_memory = cfg.PIN_MEMORY if device.type == "cuda" else False
    tr, va, te, target_ds = get_dataloaders(
        data_dir=args.data_dir, dataset_name=target, batch_size=args.batch_size,
        img_size=cfg.IMG_SIZE, num_workers=num_workers, pin_memory=pin_memory,
        random_seed=cfg.RANDOM_SEED,
    )
    target_loaders = (tr, va, te)
    target_classes = list(target_ds.classes)

    # ---- Source model ------------------------------------------------------
    source_model = None
    source_classes: List[str] = []
    if any(p in protocols for p in ("direct", "finetune")):
        ckpt_path = args.source_checkpoint
        if ckpt_path is None:
            src_dir = os.path.join(vdir, "source")
            ckpt_path = os.path.join(src_dir, "best_model.pth")
            if not (args.reuse_source and os.path.isfile(ckpt_path)):
                run_training(SimpleNamespace(
                    data_dir=args.data_dir, dataset_name=source, variant=variant,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    checkpoint_dir=src_dir, logs_dir=src_dir, results_dir=vdir,
                    output_dir=src_dir, no_pretrained=args.no_pretrained, debug=False,
                ))
        source_model, ckpt = load_source_model(ckpt_path, device)
        if source_model.variant != variant:
            raise ValueError(f"Checkpoint variant '{source_model.variant}' != requested '{variant}'")
        source_classes = list(ckpt.get("class_names") or [])
        if not source_classes:
            from dataset import load_imagefolder
            source_classes = list(load_imagefolder(args.data_dir, source, cfg.IMG_SIZE).classes)

    alignment = build_class_alignment(source_classes, target_classes) if source_classes else {}
    if source_model is not None:
        unmapped = [target_classes[i] for i in range(len(target_classes)) if i not in alignment]
        print(f"[cross_corpus] Label alignment: {len(alignment)}/{len(target_classes)} target classes mapped"
              + (f"; unmapped: {unmapped}" if unmapped else ""))
        if not alignment:
            raise RuntimeError("No source/target classes could be aligned — check folder names.")

    def record(protocol: str, y_true, y_pred, extra: dict) -> None:
        labels = list(range(len(target_classes)))
        m = compute_metrics(y_true, y_pred, labels)
        m.update({"protocol": protocol, "variant": variant, "source": source, "target": target,
                  "num_target_classes": len(target_classes),
                  "mapped_target_classes": len(alignment)}, **extra)
        write_outputs(os.path.join(vdir, protocol), y_true, y_pred, target_classes, m,
                      f"{source}→{target} [{variant}, {protocol}]")
        rows.append(m)

    # ---- Direct transfer ---------------------------------------------------
    if "direct" in protocols:
        tgt_order = sorted(alignment)
        logit_index = torch.tensor([alignment[t] for t in tgt_order], device=device)
        y_true, pos = predict(source_model, te, device, logit_index=logit_index)
        y_pred = np.array([tgt_order[p] for p in pos])
        # Samples whose true class has no source counterpart can never be right;
        # report accuracy on the mapped subset as well.
        mapped_mask = np.isin(y_true, tgt_order)
        extra = {"accuracy_mapped_only": float((y_true[mapped_mask] == y_pred[mapped_mask]).mean())
                 if mapped_mask.any() else float("nan")}
        record("direct", y_true, y_pred, extra)

    # ---- Fine-tune from source --------------------------------------------
    if "finetune" in protocols:
        set_seed(cfg.RANDOM_SEED)
        model = init_target_from_source(source_model, len(target_classes), alignment, device)
        info = train_on_target(model, target_loaders, args.finetune_epochs, args.finetune_lr,
                               os.path.join(vdir, "finetune"), device, f"{variant}/finetune")
        y_true, y_pred = predict(model, te, device)
        record("finetune", y_true, y_pred, info)
        del model

    # ---- Target-only reference --------------------------------------------
    if "scratch" in protocols:
        set_seed(cfg.RANDOM_SEED)
        model = ConvGeMNeXt(num_classes=len(target_classes), variant=variant,
                            p_init=cfg.GEM_P_INIT, dropout=cfg.DROPOUT,
                            pretrained=not args.no_pretrained).to(device)
        info = train_on_target(model, target_loaders, args.finetune_epochs, args.lr,
                               os.path.join(vdir, "scratch"), device, f"{variant}/scratch")
        y_true, y_pred = predict(model, te, device)
        record("scratch", y_true, y_pred, info)
        del model

    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = ["Variant", "Protocol", "Source", "Target", "Accuracy", "Macro_F1",
                  "Weighted_F1", "Precision", "Recall", "MCA", "Mapped_Classes", "N_Test"]


def write_summary(rows: List[dict], pair_dir: str) -> None:
    csv_path = os.path.join(pair_dir, "cross_corpus_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "Variant": r["variant"], "Protocol": r["protocol"], "Source": r["source"],
                "Target": r["target"], "Accuracy": f"{r['accuracy']:.4f}",
                "Macro_F1": f"{r['macro_f1']:.4f}", "Weighted_F1": f"{r['weighted_f1']:.4f}",
                "Precision": f"{r['precision']:.4f}", "Recall": f"{r['recall']:.4f}",
                "MCA": f"{r['mca']:.4f}", "Mapped_Classes": r["mapped_target_classes"],
                "N_Test": r["n_test"],
            })
    print(f"[cross_corpus] Summary → {csv_path}")

    variants = list(dict.fromkeys(r["variant"] for r in rows))
    protocols = list(dict.fromkeys(r["protocol"] for r in rows))
    x = np.arange(len(variants))
    width = 0.8 / max(1, len(protocols))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, p in enumerate(protocols):
        vals = [next((r["accuracy"] * 100 for r in rows if r["variant"] == v and r["protocol"] == p), 0.0)
                for v in variants]
        bars = ax.bar(x + (i - (len(protocols) - 1) / 2) * width, vals, width, label=p)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylabel("Target test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Cross-corpus — {rows[0]['source']} → {rows[0]['target']}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    png = os.path.join(pair_dir, "cross_corpus_comparison.png")
    plt.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[cross_corpus] Comparison plot → {png}")


def main(args) -> List[dict]:
    set_seed(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = cfg.resolve_dataset_name(args.source)
    target = cfg.resolve_dataset_name(args.target)
    pair_dir = os.path.join(args.results_dir, "cross_corpus", f"{source}_to_{target}")
    Path(pair_dir).mkdir(parents=True, exist_ok=True)

    variants = list(SUPPORTED_VARIANTS) if args.variant == "all" else [cfg.resolve_variant(args.variant)]
    if args.source_checkpoint and len(variants) > 1:
        raise ValueError("--source_checkpoint can only be used with a single --variant")

    print(f"\nCross-corpus: {source} → {target} | variants={variants} | protocol={args.protocol} | device={device}\n")
    t0 = time.time()
    rows: List[dict] = []
    for v in variants:
        rows.extend(run_variant(args, v, pair_dir, device))
        write_summary(rows, pair_dir)   # keep partial results if a later variant fails
    print(f"\nDone in {format_seconds(time.time() - t0)}")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-corpus training/evaluation (e.g. PlantVillage → PlantDoc).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", type=str, default="plantvillage", help="Source (training) corpus.")
    p.add_argument("--target", type=str, default="plantdoc", help="Target (evaluation) corpus.")
    p.add_argument("--variant", type=str, default="baseline",
                   help="baseline | multiscale | attention | multiscale_attention | all")
    p.add_argument("--protocol", type=str, default="all", choices=["direct", "finetune", "scratch", "all"])
    p.add_argument("--source_checkpoint", type=str, default=None,
                   help="Existing source best_model.pth (skips source training).")
    p.add_argument("--reuse_source", action="store_true",
                   help="Reuse results/cross_corpus/.../source/best_model.pth if present.")
    p.add_argument("--data_dir", type=str, default=cfg.DATA_DIR)
    p.add_argument("--results_dir", type=str, default=cfg.RESULTS_DIR)
    p.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS, help="Source training epochs.")
    p.add_argument("--finetune_epochs", type=int, default=15, help="Target fine-tune / scratch epochs.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE, help="Source / scratch learning rate.")
    p.add_argument("--finetune_lr", type=float, default=1e-4, help="Fine-tune learning rate.")
    p.add_argument("--no_pretrained", action="store_true", help="Skip ImageNet weights (offline smoke tests).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
