"""
create_sample_test_dataset.py — Synthetic SampleTest datasets (no download).

Writes ImageFolder layouts that reuse the official PlantVillage (38) and
PlantDoc (27) class folder names, with distinct synthetic leaf JPEGs.

    data/SampleTest_PlantVillage/<38 official class folders>/*.jpg
    data/SampleTest_PlantDoc/<27 official class folders>/*.jpg

Usage:
    python create_sample_test_dataset.py
    python create_sample_test_dataset.py --images_per_class 18 --force
    python create_sample_test_dataset.py --which plantdoc
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# Official PlantVillage color-dataset folder names (38 classes).
# Source: Hughes & Salathé PlantVillage; same labels as the Kaggle
# ``emmarex/plantdisease`` color split (including ``Common_rust_``).
PLANTVILLAGE_CLASSES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Official PlantDoc cropped-dataset folder names used as the 27-class
# classification benchmark (Singh et al. 2020). The extra orphan class
# ``Tomato two spotted spider mites leaf`` is omitted, matching papers
# that report 27 classes.
PLANTDOC_CLASSES: List[str] = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf",
    "Bell_pepper leaf spot",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
    "grape leaf",
    "grape leaf black rot",
]

assert len(PLANTVILLAGE_CLASSES) == 38
assert len(PLANTDOC_CLASSES) == 27
assert len(set(PLANTVILLAGE_CLASSES)) == 38
assert len(set(PLANTDOC_CLASSES)) == 27

CANONICAL_SAMPLETEST = {
    "plantvillage": "SampleTest_PlantVillage",
    "plantdoc": "SampleTest_PlantDoc",
}


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """h in [0, 360), s/v in [0, 1]."""
    h = h % 360.0
    c = v * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (
        int(max(0, min(255, round((r + m) * 255)))),
        int(max(0, min(255, round((g + m) * 255)))),
        int(max(0, min(255, round((b + m) * 255)))),
    )


def _disease_style(class_name: str) -> str:
    name = class_name.lower()
    if "healthy" in name and "unhealthy" not in name:
        return "healthy"
    # PlantDoc healthy folders are often just "<crop> leaf"
    if re.search(r"(^| )leaf$", name) and not any(
        k in name
        for k in (
            "scab", "rust", "spot", "blight", "mildew", "mold", "rot",
            "virus", "mosaic", "bacterial",
        )
    ):
        return "healthy"
    if "rust" in name:
        return "rust"
    if "mildew" in name or "mold" in name:
        return "mildew"
    if "mosaic" in name or "virus" in name or "curl" in name:
        return "virus"
    if "blight" in name or "esca" in name:
        return "blight"
    if "scab" in name or "scorch" in name:
        return "scab"
    if "bacterial" in name or "spot" in name or "septoria" in name:
        return "spots"
    if "rot" in name or "spider" in name:
        return "rot"
    return "spots"


def make_leaf_image(
    class_idx: int,
    n_classes: int,
    class_name: str,
    seed: int,
    size: int = 224,
) -> Image.Image:
    """Synthetic leaf: unique hue + disease texture so classes stay separable."""
    rng = np.random.default_rng(seed)
    style = _disease_style(class_name)

    # Spread hues evenly; keep saturation/value class-specific.
    hue = (class_idx * 360.0 / n_classes + rng.uniform(-4, 4)) % 360.0
    sat = 0.35 + 0.45 * ((class_idx * 5) % 7) / 6.0
    val = 0.40 + 0.40 * ((class_idx * 3) % 5) / 4.0
    if style == "healthy":
        hue = 95.0 + (class_idx % 12) * 4.0
        sat = 0.55 + (class_idx % 5) * 0.06
        val = 0.55 + (class_idx % 4) * 0.08

    leaf_color = _hsv_to_rgb(hue, sat, val)
    bg_color = _hsv_to_rgb((hue + 40) % 360, 0.25, 0.18)
    vein_color = tuple(max(0, c - 45) for c in leaf_color)

    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = np.array(bg_color, dtype=np.uint8)
    # faint background grain
    noise = rng.integers(-8, 9, size=(size, size, 1), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    cx = size // 2 + int(rng.integers(-10, 11))
    cy = size // 2 + int(rng.integers(-10, 11))
    rx = int(size * (0.32 + 0.08 * ((class_idx % 4) / 3.0)))
    ry = int(size * (0.20 + 0.10 * (((class_idx + 1) % 5) / 4.0)))
    jitter = tuple(int(rng.integers(-12, 13)) for _ in range(3))
    fill = tuple(int(max(0, min(255, c + j))) for c, j in zip(leaf_color, jitter))

    draw.ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry],
        fill=fill,
        outline=tuple(max(0, c - 50) for c in fill),
        width=3,
    )

    n_veins = 2 + (class_idx % 5)
    for k in range(n_veins):
        x0 = cx + int((k - n_veins / 2) * rx / max(n_veins, 1))
        draw.line(
            [(x0, cy - ry + 4), (cx, cy + ry - 4)],
            fill=vein_color,
            width=1 + (class_idx % 3 == 0),
        )

    def _spot(color: Tuple[int, int, int], n: int, rmin: int, rmax: int) -> None:
        for _ in range(n):
            t = float(rng.uniform(0, 2 * math.pi))
            rad = float(rng.uniform(0, 1)) ** 0.5
            sx = int(cx + rad * rx * 0.85 * math.cos(t))
            sy = int(cy + rad * ry * 0.85 * math.sin(t))
            sr = int(rng.integers(rmin, rmax + 1))
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=color)

    if style == "healthy":
        _spot(tuple(min(255, c + 20) for c in fill), n=2, rmin=2, rmax=4)
    elif style == "rust":
        _spot(_hsv_to_rgb(28, 0.85, 0.75), n=18, rmin=2, rmax=6)
    elif style == "mildew":
        _spot((230, 230, 235), n=22, rmin=3, rmax=8)
    elif style == "virus":
        # patchy mosaic rectangles
        for _ in range(10):
            w = int(rng.integers(8, 22))
            h = int(rng.integers(8, 22))
            sx = int(rng.integers(cx - rx, cx + rx - w))
            sy = int(rng.integers(cy - ry, cy + ry - h))
            mosaic = _hsv_to_rgb((hue + 80) % 360, 0.7, 0.7)
            draw.rectangle([sx, sy, sx + w, sy + h], fill=mosaic)
    elif style == "blight":
        _spot((90, 50, 20), n=8, rmin=6, rmax=14)
        _spot((40, 20, 10), n=6, rmin=3, rmax=7)
    elif style == "scab":
        _spot((70, 55, 30), n=14, rmin=3, rmax=7)
    elif style == "rot":
        _spot((20, 15, 15), n=10, rmin=5, rmax=12)
    else:  # spots / bacterial
        _spot((40, 20, 20), n=16, rmin=2, rmax=5)
        _spot((160, 40, 40), n=8, rmin=2, rmax=4)

    # class-index ring so even similar diseases stay distinct
    ring = _hsv_to_rgb((hue + 180) % 360, 0.8, 0.9)
    margin = 6 + (class_idx % 5)
    draw.rectangle(
        [margin, margin, size - margin, size - margin],
        outline=ring,
        width=1 + (class_idx % 3),
    )

    img = img.filter(ImageFilter.GaussianBlur(radius=0.35))
    return img


def _safe_stem(class_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", class_name).strip("_")
    return stem.lower() or "class"


def _count_jpegs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.jpg"))


def write_synthetic_dataset(
    data_dir: str,
    dataset_name: str,
    class_names: Sequence[str],
    images_per_class: int = 18,
    img_size: int = 224,
    force: bool = False,
) -> str:
    """Write one ImageFolder dataset. Returns the dataset root path."""
    root = Path(data_dir) / dataset_name
    if root.exists() and not force:
        existing = _count_jpegs(root)
        if existing > 0:
            print(f"[sample-test] Already exists: {root} ({existing} images)")
            print("[sample-test] Use --force to regenerate.")
            return str(root)

    if force and root.exists():
        shutil.rmtree(root)

    print(f"[sample-test] Creating {dataset_name} at {root}/")
    n_classes = len(class_names)
    for cls_idx, class_name in enumerate(class_names):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_stem(class_name)
        for i in range(images_per_class):
            img = make_leaf_image(
                class_idx=cls_idx,
                n_classes=n_classes,
                class_name=class_name,
                seed=cls_idx * 10_000 + i,
                size=img_size,
            )
            img.save(class_dir / f"{stem}_{i:03d}.jpg", quality=88)

    total = images_per_class * n_classes
    print(
        f"[sample-test] Done — {n_classes} classes × {images_per_class} "
        f"images = {total} total → {root}"
    )
    return str(root)


def create_sample_test_datasets(
    data_dir: str = "data",
    images_per_class: int = 18,
    img_size: int = 224,
    force: bool = False,
    which: str = "both",
) -> List[str]:
    """Generate SampleTest_PlantVillage and/or SampleTest_PlantDoc."""
    which = which.lower()
    created: List[str] = []
    if which in ("both", "plantvillage"):
        created.append(
            write_synthetic_dataset(
                data_dir=data_dir,
                dataset_name=CANONICAL_SAMPLETEST["plantvillage"],
                class_names=PLANTVILLAGE_CLASSES,
                images_per_class=images_per_class,
                img_size=img_size,
                force=force,
            )
        )
    if which in ("both", "plantdoc"):
        created.append(
            write_synthetic_dataset(
                data_dir=data_dir,
                dataset_name=CANONICAL_SAMPLETEST["plantdoc"],
                class_names=PLANTDOC_CLASSES,
                images_per_class=images_per_class,
                img_size=img_size,
                force=force,
            )
        )
    if which not in ("both", "plantvillage", "plantdoc"):
        raise ValueError("--which must be plantvillage, plantdoc, or both")
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic SampleTest PlantVillage (38) and PlantDoc (27) datasets.",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--images_per_class", type=int, default=18)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--which",
        type=str,
        default="both",
        choices=["both", "plantvillage", "plantdoc"],
        help="Which SampleTest dataset(s) to write.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if images exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_sample_test_datasets(
        data_dir=args.data_dir,
        images_per_class=args.images_per_class,
        img_size=args.img_size,
        force=args.force,
        which=args.which,
    )


if __name__ == "__main__":
    main()
