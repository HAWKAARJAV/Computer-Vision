"""
prepare_datasets.py — Download, arrange and verify the datasets for every experiment
====================================================================================
Produces the on-disk layout the experiments expect:

  data/PlantVillage/<38 class folders>/*.jpg          (Kaggle: abdallahalidev/plantvillage-dataset, "color")
  data/PlantDoc/train/<class>/*.jpg                   (GitHub: pratikkayal/PlantDoc-Dataset)
  data/PlantDoc/test/<class>/*.jpg
  data/PlantVillage_split_seed42.json                 (70/15/15 stratified split, reused by every run)
  data/PlantDoc_split_seed42.json                     (official test set kept; val carved from train/)
  data/dataset_report.md                              (per-class counts, totals, checks vs the paper)

Usage
-----
  # 1. Download + arrange both datasets (needs the Kaggle CLI + ~/.kaggle/kaggle.json, and git)
  python prepare_datasets.py --plantvillage --plantdoc

  # ...or point at copies you already downloaded (zip or extracted folder)
  python prepare_datasets.py --plantvillage --pv_source ~/Downloads/plantvillage-dataset.zip
  python prepare_datasets.py --plantdoc --pd_source ~/Downloads/PlantDoc-Dataset

  # 2. Verify counts and create the reusable split files
  python prepare_datasets.py --verify

  # Tiny random-pixel copy with the real class names, for a CPU pipeline check
  python prepare_datasets.py --make_smoke          # → data_smoke/PlantVillage, data_smoke/PlantDoc
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DOWNLOADS = DATA / "_downloads"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

PV_KAGGLE = "abdallahalidev/plantvillage-dataset"
PD_GIT = "https://github.com/pratikkayal/PlantDoc-Dataset.git"

# Paper reference values (Arshad et al., 2026, Tables 3–5)
PAPER_PV_IMAGES, PAPER_PV_CLASSES = 54_284, 38
PAPER_PD_CLASSES = 27

PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]


def count_images(folder: Path) -> int:
    return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)


def class_dirs(folder: Path) -> list[Path]:
    return sorted(d for d in folder.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def place(src: Path, dst: Path, move: bool) -> None:
    if dst.exists():
        raise SystemExit(f"{dst} already exists — remove it first or use --force.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"{'Moving' if move else 'Copying'} {src} → {dst}")
    shutil.move(str(src), str(dst)) if move else shutil.copytree(src, dst)


def extract_if_zip(src: Path, name: str) -> Path:
    if src.is_file() and src.suffix == ".zip":
        out = DOWNLOADS / name
        print(f"Extracting {src} → {out}")
        with zipfile.ZipFile(src) as z:
            z.extractall(out)
        return out
    return src


# ---------------------------------------------------------------------------
# PlantVillage
# ---------------------------------------------------------------------------

def find_plantvillage_root(base: Path) -> Path:
    """Find the folder holding the 38 class folders (prefers the 'color' variant)."""
    candidates = []
    for d, subdirs, _ in os.walk(base):
        hits = sum(1 for s in subdirs if "___" in s)
        if hits >= 30:
            candidates.append(Path(d))
    if not candidates:
        raise SystemExit(f"No folder with PlantVillage class sub-folders found under {base}")
    candidates.sort(key=lambda p: (0 if "color" in p.name.lower() else 1, len(p.parts)))
    return candidates[0]


def prepare_plantvillage(source: str | None, move: bool, force: bool) -> None:
    dst = DATA / "PlantVillage"
    if force and dst.exists():
        shutil.rmtree(dst)
    if source is None:
        if shutil.which("kaggle") is None:
            raise SystemExit(
                "Kaggle CLI not found. Either:\n"
                "  pip install kaggle  and put your API token in ~/.kaggle/kaggle.json, or\n"
                f"  download https://www.kaggle.com/datasets/{PV_KAGGLE} manually and pass --pv_source <zip|folder>"
            )
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        run(["kaggle", "datasets", "download", "-d", PV_KAGGLE, "-p", str(DOWNLOADS / "plantvillage"), "--unzip"])
        base = DOWNLOADS / "plantvillage"
    else:
        base = extract_if_zip(Path(source).expanduser(), "plantvillage")
    root = find_plantvillage_root(base)
    print(f"PlantVillage class root: {root}")
    place(root, dst, move)


# ---------------------------------------------------------------------------
# PlantDoc
# ---------------------------------------------------------------------------

def prepare_plantdoc(source: str | None, move: bool, force: bool) -> None:
    dst = DATA / "PlantDoc"
    if force and dst.exists():
        shutil.rmtree(dst)
    if source is None:
        repo = DOWNLOADS / "PlantDoc-Dataset"
        if not repo.exists():
            DOWNLOADS.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--depth", "1", PD_GIT, str(repo)])
        base = repo
    else:
        base = extract_if_zip(Path(source).expanduser(), "plantdoc")
    hits = [p for p in base.rglob("train") if p.is_dir() and (p.parent / "test").is_dir()]
    if not hits:
        raise SystemExit(f"No train/ + test/ folders found under {base}")
    root = hits[0].parent
    print(f"PlantDoc root: {root}")
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        place(root / split, dst / split, move)


# ---------------------------------------------------------------------------
# Verification + split files
# ---------------------------------------------------------------------------

def verify(data_dir: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from cross_corpus import PD_TO_PV_CLASS_MAP
    from dataset import get_dataloaders

    lines = ["# Dataset report", ""]
    ok = True

    pv = data_dir / "PlantVillage"
    if pv.is_dir():
        counts = {d.name: count_images(d) for d in class_dirs(pv)}
        total = sum(counts.values())
        missing = sorted(set(PLANTVILLAGE_CLASSES) - set(counts))
        lines += ["## PlantVillage", "",
                  f"- Classes: **{len(counts)}** (paper: {PAPER_PV_CLASSES})",
                  f"- Images: **{total:,}** (paper: {PAPER_PV_IMAGES:,})",
                  f"- Images per class: {min(counts.values())} – {max(counts.values())}",
                  f"- Missing official classes: {missing or 'none'}", "",
                  "| Class | Images |", "|---|---|"] + [f"| {k} | {v} |" for k, v in counts.items()] + [""]
        ok &= len(counts) == PAPER_PV_CLASSES and not missing
    else:
        lines += ["## PlantVillage", "", "- **Not found** — run `python prepare_datasets.py --plantvillage`", ""]
        ok = False

    pd = data_dir / "PlantDoc"
    if (pd / "train").is_dir() and (pd / "test").is_dir():
        tr = {d.name: count_images(d) for d in class_dirs(pd / "train")}
        te = {d.name: count_images(d) for d in class_dirs(pd / "test")}
        shared = sorted(set(tr) & set(te))
        unmapped = [c for c in shared if c not in PD_TO_PV_CLASS_MAP]
        lines += ["## PlantDoc", "",
                  f"- Classes in both train/ and test/: **{len(shared)}** (paper: {PAPER_PD_CLASSES})",
                  f"- Dropped (only in one split): {sorted(set(tr) ^ set(te)) or 'none'}",
                  f"- Images: train **{sum(tr[c] for c in shared):,}**, test **{sum(te[c] for c in shared):,}**",
                  f"- Classes without a PlantVillage mapping (cross-corpus): {unmapped or 'none'}", "",
                  "| Class | Train | Test | PlantVillage counterpart |", "|---|---|---|---|"]
        lines += [f"| {c} | {tr[c]} | {te[c]} | {PD_TO_PV_CLASS_MAP.get(c, '—')} |" for c in shared] + [""]
        ok &= len(shared) == PAPER_PD_CLASSES and not unmapped
    else:
        lines += ["## PlantDoc", "", "- **Not found** — run `python prepare_datasets.py --plantdoc`", ""]
        ok = False

    lines += ["## Splits (seed 42)", ""]
    for name in ("PlantVillage", "PlantDoc"):
        if (data_dir / name).is_dir():
            tr_l, va_l, te_l, _ = get_dataloaders(str(data_dir), name, batch_size=8, num_workers=0, pin_memory=False)
            lines.append(f"- {name}: train {len(tr_l.dataset):,} / val {len(va_l.dataset):,} / "
                         f"test {len(te_l.dataset):,} → `{name}_split_seed42.json`")
    lines.append("")
    lines.append(f"**Status: {'READY' if ok else 'INCOMPLETE — see above'}**")

    report = data_dir / "dataset_report.md"
    report.write_text("\n".join(lines))
    print("\n".join(l for l in lines if not l.startswith("| ")))
    print(f"\nReport → {report}")


# ---------------------------------------------------------------------------
# Smoke-test data
# ---------------------------------------------------------------------------

def make_smoke(out: Path, per_class: int) -> None:
    """Random-pixel images with the real class names — checks the pipeline, not accuracy."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT))
    from cross_corpus import PD_TO_PV_CLASS_MAP

    rng = np.random.default_rng(0)

    def save(folder: Path, n: int, seed: int) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        base = np.array([(seed * 37) % 255, 110 + seed % 90, (seed * 53) % 255])
        for i in range(n):
            arr = (rng.random((64, 64, 3)) * 70 + base).clip(0, 255).astype("uint8")
            Image.fromarray(arr).save(folder / f"{i:03d}.jpg")

    for i, c in enumerate(PLANTVILLAGE_CLASSES):
        save(out / "PlantVillage" / c, per_class, i)
    pd_classes = [c for c in PD_TO_PV_CLASS_MAP if c != "Tomato two spotted spider mites leaf"]
    for c in pd_classes:
        seed = PLANTVILLAGE_CLASSES.index(PD_TO_PV_CLASS_MAP[c])
        save(out / "PlantDoc" / "train" / c, per_class, seed)
        save(out / "PlantDoc" / "test" / c, max(2, per_class // 3), seed)
    save(out / "PlantDoc" / "train" / "Tomato two spotted spider mites leaf", 2, 99)  # orphan, like the real set
    print(f"Smoke datasets → {out}  (random pixels; use only to check the pipeline runs)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plantvillage", action="store_true", help="Download/arrange PlantVillage.")
    p.add_argument("--plantdoc", action="store_true", help="Download/arrange PlantDoc.")
    p.add_argument("--pv_source", default=None, help="Existing PlantVillage zip or folder (skips Kaggle download).")
    p.add_argument("--pd_source", default=None, help="Existing PlantDoc zip or folder (skips git clone).")
    p.add_argument("--move", action="store_true", help="Move instead of copy (saves disk space).")
    p.add_argument("--force", action="store_true", help="Replace existing data/PlantVillage or data/PlantDoc.")
    p.add_argument("--verify", action="store_true", help="Check counts and create split files.")
    p.add_argument("--data_dir", default=str(DATA))
    p.add_argument("--make_smoke", action="store_true", help="Create tiny random-pixel datasets in data_smoke/.")
    p.add_argument("--smoke_per_class", type=int, default=10)
    a = p.parse_args()

    if not any([a.plantvillage, a.plantdoc, a.verify, a.make_smoke]):
        p.print_help()
        return
    if a.plantvillage:
        prepare_plantvillage(a.pv_source, a.move, a.force)
    if a.plantdoc:
        prepare_plantdoc(a.pd_source, a.move, a.force)
    if a.make_smoke:
        make_smoke(ROOT / "data_smoke", a.smoke_per_class)
    if a.verify:
        verify(Path(a.data_dir))


if __name__ == "__main__":
    main()
