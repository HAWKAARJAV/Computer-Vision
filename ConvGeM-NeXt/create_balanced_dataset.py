"""
create_balanced_dataset.py — Medium-difficulty synthetic PlantVillage-style set.

Between SampleTest (very separable hues) and RealisticPV (heavy overlap).
Leaf-on-soil look suitable for a viva demo; each class has learnable crop hue
bands plus distinct (but not cartoon) symptom patterns. Intra-class variation
(rotation, brightness, mild blur) is moderate — less than RealisticPV.

    data/BalancedPV/<38 official PlantVillage class folders>/*.jpg

Recommended training (RTX 4050, four-variant ablation ~1–2 h):
    python ablation_study.py --dataset balanced --epochs 12 --batch_size 32

Usage:
    python create_balanced_dataset.py
    python create_balanced_dataset.py --images_per_class 32 --force
    python create_balanced_dataset.py --skip_knn
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from create_realistic_dataset import (
    _Symptom,
    _blend_gaussian,
    _crop_key,
    _hsv_to_rgb,
    _paint_symptom,
    _sym,
    _xy,
)
from create_sample_test_dataset import PLANTVILLAGE_CLASSES


CANONICAL_NAME = "BalancedPV"
DEFAULT_IMAGES_PER_CLASS = 32
DEFAULT_IMG_SIZE = 224

# Crop tint bands (learnable, ~45° spread — not the full SampleTest hue wheel).
_CROP_HUE: Dict[str, float] = {
    "apple": 86.0,
    "blueberry": 112.0,
    "cherry": 94.0,
    "corn": 122.0,
    "grape": 80.0,
    "orange": 108.0,
    "peach": 92.0,
    "pepper": 84.0,
    "potato": 118.0,
    "raspberry": 90.0,
    "soybean": 120.0,
    "squash": 104.0,
    "strawberry": 82.0,
    "tomato": 78.0,
}
_CROP_SAT_BIAS: Dict[str, float] = {
    "apple": 0.02,
    "blueberry": -0.04,
    "cherry": 0.0,
    "corn": 0.05,
    "grape": -0.02,
    "orange": 0.03,
    "peach": 0.01,
    "pepper": 0.04,
    "potato": -0.03,
    "raspberry": 0.0,
    "soybean": 0.03,
    "squash": 0.02,
    "strawberry": -0.01,
    "tomato": 0.02,
}

# Per-class lesion recipes — narrower ranges than RealisticPV; related diseases differ visibly.
BALANCED_SYMPTOMS: Dict[str, _Symptom] = {
    "Apple___Apple_scab": _sym(
        (10, 16), (2.2, 3.8), (0.42, 0.34, 0.16), 0.50, elongate=1.25, ring=0.25, edge=0.15,
    ),
    "Apple___Black_rot": _sym(
        (3, 6), (5.0, 8.5), (0.16, 0.10, 0.08), 0.60, elongate=1.15, ring=0.30,
    ),
    "Apple___Cedar_apple_rust": _sym(
        (8, 14), (2.4, 4.2), (0.74, 0.44, 0.12), 0.54, elongate=1.0, ring=0.40, kind="rust",
    ),
    "Cherry_(including_sour)___Powdery_mildew": _sym(
        (16, 24), (2.4, 4.0), (0.84, 0.84, 0.80), 0.34, elongate=1.05, kind="mildew",
    ),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": _sym(
        (6, 10), (2.6, 4.0), (0.48, 0.44, 0.34), 0.50, elongate=2.6, ring=0.12,
    ),
    "Corn_(maize)___Common_rust_": _sym(
        (16, 24), (1.6, 2.8), (0.72, 0.42, 0.12), 0.52, elongate=1.05, ring=0.22, kind="rust",
    ),
    "Corn_(maize)___Northern_Leaf_Blight": _sym(
        (3, 5), (3.5, 5.0), (0.58, 0.42, 0.18), 0.52, elongate=4.0, kind="streak",
    ),
    "Grape___Black_rot": _sym(
        (5, 9), (2.8, 5.0), (0.18, 0.12, 0.10), 0.56, elongate=1.0, ring=0.55,
    ),
    "Grape___Esca_(Black_Measles)": _sym(
        (4, 7), (7.0, 11.0), (0.68, 0.62, 0.26), 0.38, elongate=2.0,
        hue_shift=(-12.0, -6.0), kind="mosaic",
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": _sym(
        (8, 12), (3.2, 5.5), (0.48, 0.28, 0.14), 0.50, elongate=1.2, ring=0.28,
    ),
    "Orange___Haunglongbing_(Citrus_greening)": _sym(
        (4, 7), (9.0, 14.0), (0.76, 0.70, 0.28), 0.42, elongate=1.35,
        hue_shift=(-14.0, -8.0), kind="mosaic",
    ),
    "Peach___Bacterial_spot": _sym(
        (14, 22), (1.4, 2.4), (0.34, 0.14, 0.10), 0.56, elongate=1.02,
    ),
    "Pepper,_bell___Bacterial_spot": _sym(
        (12, 18), (1.6, 2.6), (0.40, 0.18, 0.12), 0.54, elongate=1.06,
    ),
    "Potato___Early_blight": _sym(
        (8, 14), (3.0, 5.5), (0.50, 0.32, 0.16), 0.52, elongate=1.12, ring=0.48,
    ),
    "Potato___Late_blight": _sym(
        (4, 8), (6.0, 9.5), (0.30, 0.16, 0.09), 0.54, elongate=1.22, ring=0.15,
    ),
    "Squash___Powdery_mildew": _sym(
        (18, 26), (2.4, 4.5), (0.86, 0.86, 0.82), 0.38, elongate=1.1, kind="mildew",
    ),
    "Strawberry___Leaf_scorch": _sym(
        (9, 14), (2.8, 4.8), (0.46, 0.24, 0.12), 0.52, elongate=1.25, ring=0.18, edge=0.75,
    ),
    "Tomato___Bacterial_spot": _sym(
        (16, 24), (1.2, 2.2), (0.32, 0.12, 0.09), 0.58, elongate=1.0,
    ),
    "Tomato___Early_blight": _sym(
        (7, 12), (3.4, 6.0), (0.52, 0.34, 0.17), 0.52, elongate=1.08, ring=0.52,
    ),
    "Tomato___Late_blight": _sym(
        (4, 8), (5.5, 9.0), (0.32, 0.18, 0.10), 0.54, elongate=1.25, ring=0.20,
    ),
    "Tomato___Leaf_Mold": _sym(
        (5, 9), (5.0, 9.0), (0.66, 0.60, 0.28), 0.40, elongate=1.3,
        hue_shift=(-9.0, -4.0), kind="mildew",
    ),
    "Tomato___Septoria_leaf_spot": _sym(
        (12, 18), (1.8, 3.2), (0.42, 0.24, 0.12), 0.54, elongate=1.02, ring=0.70,
        light_center=True,
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": _sym(
        (32, 48), (0.8, 1.4), (0.78, 0.74, 0.48), 0.40, elongate=1.0, kind="stipple",
    ),
    "Tomato___Target_Spot": _sym(
        (8, 12), (2.8, 5.0), (0.52, 0.34, 0.17), 0.52, elongate=1.02, ring=0.78,
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": _sym(
        (4, 7), (7.0, 11.0), (0.76, 0.72, 0.28), 0.34, elongate=1.45,
        hue_shift=(-14.0, -7.0), kind="mosaic",
    ),
    "Tomato___Tomato_mosaic_virus": _sym(
        (6, 10), (5.5, 9.5), (0.64, 0.68, 0.26), 0.36, elongate=1.25,
        hue_shift=(-5.0, 0.0), kind="mosaic",
    ),
}

_HEALTHY_DUST: _Symptom = _sym(
    (0, 2), (1.0, 1.8), (0.38, 0.30, 0.18), 0.18, elongate=1.05, kind="spots",
)


def _assert_catalog() -> None:
    missing = [
        name for name in PLANTVILLAGE_CLASSES
        if "healthy" not in name.lower() and name not in BALANCED_SYMPTOMS
    ]
    extra = [name for name in BALANCED_SYMPTOMS if name not in PLANTVILLAGE_CLASSES]
    if missing or extra or len(PLANTVILLAGE_CLASSES) != 38:
        raise RuntimeError(f"Balanced symptom catalog mismatch. missing={missing} extra={extra}")


_assert_catalog()


def _family_hue_shift(class_name: str) -> float:
    """Disease-family tint on top of crop band (related classes may overlap slightly)."""
    low = class_name.lower()
    if "healthy" in low:
        return 5.0
    if "rust" in low:
        return 12.0
    if "mildew" in low or "mold" in low:
        return -8.0
    if "mosaic" in low or "virus" in low or "curl" in low:
        return 15.0
    if "blight" in low or "esca" in low:
        return 7.0
    if "bacterial" in low or "septoria" in low:
        return -5.0
    if "scab" in low or "scorch" in low:
        return 3.0
    if "spider" in low or "stipple" in low:
        return 10.0
    if "spot" in low or "rot" in low:
        return -2.0
    return 0.0


def _build_class_hues() -> Dict[str, float]:
    """Crop band + disease family + class index spread (~110° total, not a full wheel)."""
    n = len(PLANTVILLAGE_CLASSES)
    out: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for class_idx, class_name in enumerate(PLANTVILLAGE_CLASSES):
        crop = _crop_key(class_name)
        slot = counts.get(crop, 0)
        counts[crop] = slot + 1
        n_crop = sum(1 for c in PLANTVILLAGE_CLASSES if _crop_key(c) == crop)
        slot_offset = (slot - (n_crop - 1) / 2.0) * 3.5
        index_spread = 78.0 + (class_idx * 142.0 / max(n - 1, 1))
        crop_pull = 0.28 * (_CROP_HUE.get(crop, 100.0) - 100.0)
        out[class_name] = (
            index_spread
            + crop_pull
            + _family_hue_shift(class_name)
            + slot_offset
        )
    return out


_CLASS_TARGET_HUE: Dict[str, float] = _build_class_hues()


def _boost_symptom(symptom: _Symptom) -> _Symptom:
    out = dict(symptom)
    out["alpha"] = float(np.clip(float(symptom["alpha"]) * 1.32, 0.14, 0.85))
    return out  # type: ignore[return-value]


def _sample_leaf_color(
    rng: np.random.Generator,
    class_name: str,
    symptom: _Symptom,
    class_idx: int,
) -> np.ndarray:
    crop = _crop_key(class_name)
    shift_lo, shift_hi = symptom["hue_shift"]  # type: ignore[misc]
    hue = float(
        _CLASS_TARGET_HUE[class_name]
        + rng.normal(0.0, 2.8)
        + rng.uniform(shift_lo, shift_hi)
    )
    sat_bias = _CROP_SAT_BIAS.get(crop, 0.0)
    if "healthy" in class_name.lower():
        sat = float(np.clip(rng.normal(0.60 + sat_bias, 0.05), 0.46, 0.74))
        val = float(np.clip(rng.normal(0.60, 0.05), 0.44, 0.74))
    else:
        sat = float(np.clip(rng.normal(0.52 + sat_bias + (class_idx % 6) * 0.012, 0.05), 0.38, 0.70))
        val = float(np.clip(rng.normal(0.56 + (class_idx % 5) * 0.014, 0.05), 0.40, 0.72))
    return _hsv_to_rgb(hue, sat, val)


def make_balanced_leaf_image(
    class_name: str,
    class_idx: int,
    seed: int,
    size: int = DEFAULT_IMG_SIZE,
) -> Image.Image:
    """Synthetic leaf on soil — medium difficulty, no painted labels."""
    rng = np.random.default_rng(seed)
    is_healthy = "healthy" in class_name.lower()
    symptom = _HEALTHY_DUST if is_healthy else BALANCED_SYMPTOMS[class_name]
    yy, xx = _xy(size)

    soil = np.array([0.42, 0.31, 0.21], dtype=np.float32)
    soil = np.clip(soil * np.float32(rng.uniform(0.88, 1.10)) + rng.normal(0, 0.012, 3), 0, 1)
    img = np.broadcast_to(soil, (size, size, 3)).astype(np.float32).copy()

    coarse = rng.normal(0, 1, (12, 12)).astype(np.float32)
    coarse_img = Image.fromarray(np.clip(coarse * 26 + 128, 0, 255).astype(np.uint8), mode="L")
    coarse_big = np.asarray(coarse_img.resize((size, size), Image.BILINEAR), dtype=np.float32)
    img += ((coarse_big - 128.0) / 128.0 * 0.045)[..., None]
    img += rng.normal(0, 0.010, (size, size, 1)).astype(np.float32)

    for _ in range(int(rng.integers(5, 11))):
        px = float(rng.uniform(0, size - 1))
        py = float(rng.uniform(0, size - 1))
        pr = float(rng.uniform(1.6, 4.8))
        pebble = np.clip(soil * np.float32(rng.uniform(0.70, 1.25)), 0, 1)
        _blend_gaussian(
            img, px, py, pr, pr * float(rng.uniform(0.65, 1.25)),
            float(rng.uniform(0, math.pi)), pebble, float(rng.uniform(0.30, 0.60)),
        )

    ang = float(rng.uniform(-8.0, 8.0) * math.pi / 180.0)
    scale = float(rng.uniform(0.96, 1.04))
    rx = size * (0.34 + 0.02 * (class_idx % 5) / 4.0) * scale
    ry = size * (0.46 + 0.02 * ((class_idx + 2) % 6) / 5.0) * scale
    cx = size * 0.5 + float(rng.uniform(-10, 10))
    cy = size * 0.5 + float(rng.uniform(-8, 8))
    ca = math.cos(ang)
    sa = math.sin(ang)
    dx = xx - np.float32(cx)
    dy = yy - np.float32(cy)
    xr = ca * dx + sa * dy
    yr = -sa * dx + ca * dy
    rnorm = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    mask = np.clip((1.02 - rnorm) / 0.035, 0.0, 1.0).astype(np.float32)

    leaf_color = _sample_leaf_color(rng, class_name, symptom, class_idx)
    mottling_small = rng.normal(0, 1, (10, 10, 3)).astype(np.float32)
    mottling_u8 = np.clip(mottling_small * 28 + 128, 0, 255).astype(np.uint8)
    mottling = np.asarray(
        Image.fromarray(mottling_u8, mode="RGB").resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    )
    mottling = (mottling - 128.0) / 128.0 * 0.038
    leaf_rgb = np.clip(leaf_color.reshape(1, 1, 3) + mottling, 0, 1)
    img = img * (1.0 - mask[..., None]) + leaf_rgb * mask[..., None]

    vein = np.exp(-0.5 * (xr / 1.15) ** 2) * (np.abs(yr) < ry * 0.92)
    for side in (-1.0, 1.0):
        for t, spread in ((-0.55, 0.20), (-0.22, 0.14), (0.12, 0.16), (0.42, 0.18)):
            y0 = (t + float(rng.uniform(-0.02, 0.02))) * ry
            x1 = side * rx * 0.78
            y1 = y0 + side * float(rng.uniform(-spread, spread)) * ry * 0.12 + spread * ry * 0.04
            vx, vy = x1 - 0.0, y1 - y0
            length2 = vx * vx + vy * vy + 1e-6
            proj = np.clip((xr * vx + (yr - y0) * vy) / length2, 0.0, 1.0)
            dist = np.hypot(xr - proj * vx, (yr - y0) - proj * vy)
            vein = np.maximum(vein, np.exp(-0.5 * (dist / 0.85) ** 2) * (proj > 0) * (proj < 1))
    vein *= mask
    darken = (0.09 + float(rng.uniform(-0.015, 0.015))) * vein[..., None]
    img = np.clip(img - darken, 0, 1)

    if not is_healthy:
        _paint_symptom(img, mask, cx, cy, rx, ry, ang, _boost_symptom(symptom), rng)

    if rng.random() < 0.22:
        for _ in range(int(rng.integers(1, 2))):
            theta = float(rng.uniform(0, 2 * math.pi))
            ox = cx + math.cos(theta) * rx * float(rng.uniform(0.78, 1.08))
            oy = cy + math.sin(theta) * ry * float(rng.uniform(0.78, 1.08))
            clod = np.clip(soil * np.float32(rng.uniform(0.78, 1.12)), 0, 1)
            _blend_gaussian(
                img, ox, oy,
                float(rng.uniform(5, 11)), float(rng.uniform(4, 9)),
                float(rng.uniform(0, math.pi)), clod, float(rng.uniform(0.40, 0.65)),
            )

    if rng.random() < 0.45:
        shade_ang = float(rng.uniform(0, 2 * math.pi))
        ramp = xx * math.cos(shade_ang) + yy * math.sin(shade_ang)
        ramp = (ramp - ramp.min()) / (float(ramp.max() - ramp.min()) + 1e-6)
        cut = float(rng.uniform(0.32, 0.58))
        soft = np.clip((ramp - cut) / 0.30, 0.0, 1.0).astype(np.float32)
        darkness = float(rng.uniform(0.68, 0.88))
        img *= (1.0 - soft[..., None] * (1.0 - darkness))

    gain = float(rng.uniform(0.88, 1.12))
    contrast = float(rng.uniform(0.88, 1.12))
    mid = float(np.median(img))
    img = (img - mid) * contrast + mid * gain
    img = np.clip(img, 0.0, 1.0)
    gamma = float(rng.uniform(0.92, 1.10))
    img = np.power(img, gamma)

    u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB")
    if rng.random() < 0.14:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.25, 0.50))))
    rot = float(rng.uniform(-6.0, 6.0))
    if abs(rot) > 0.5:
        pil = pil.rotate(rot, resample=Image.BILINEAR, fillcolor=(int(soil[0] * 255), int(soil[1] * 255), int(soil[2] * 255)))
    return pil


def _count_jpegs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.jpg"))


def create_balanced_dataset(
    data_dir: str = "data",
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
    img_size: int = DEFAULT_IMG_SIZE,
    force: bool = False,
    dataset_name: str = CANONICAL_NAME,
) -> str:
    """Write ``data/BalancedPV``. Returns the dataset root path."""
    if images_per_class < 1:
        raise ValueError("images_per_class must be >= 1")
    if dataset_name != CANONICAL_NAME:
        raise ValueError(f"This generator only writes {CANONICAL_NAME}.")

    root = Path(data_dir) / dataset_name
    if root.exists() and not force:
        existing = _count_jpegs(root)
        if existing > 0:
            print(f"[balanced] Already exists: {root} ({existing} images)")
            print("[balanced] Use --force to regenerate.")
            return str(root)

    if force and root.exists():
        shutil.rmtree(root)

    print(f"[balanced] Creating {dataset_name} at {root}/")
    print(
        "[balanced] Medium difficulty — crop hue bands + distinct symptoms; "
        f"{len(PLANTVILLAGE_CLASSES)} classes × {images_per_class} images."
    )
    for cls_idx, class_name in enumerate(PLANTVILLAGE_CLASSES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = make_balanced_leaf_image(
                class_name=class_name,
                class_idx=cls_idx,
                seed=30_260 + cls_idx * 10_000 + i,
                size=img_size,
            )
            img.save(class_dir / f"{i:03d}.jpg", quality=85, subsampling=2)
        print(f"[balanced]   {cls_idx + 1:02d}/38  {class_name}")

    total = images_per_class * len(PLANTVILLAGE_CLASSES)
    print(f"[balanced] Done — {len(PLANTVILLAGE_CLASSES)} classes × {images_per_class} = {total} → {root}")
    return str(root)


def knn_sanity_check(
    dataset_root: str | Path,
    *,
    downsample: int = 32,
    k: int = 5,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> float:
    """Histogram + downsampled-pixel kNN hold-out accuracy (no CNN)."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    root = Path(dataset_root)
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    features: List[np.ndarray] = []
    labels: List[int] = []
    for class_name in class_names:
        cls_idx = name_to_idx[class_name]
        for img_path in sorted((root / class_name).glob("*.jpg")):
            im = Image.open(img_path).convert("RGB")
            w, h = im.size
            margin = int(min(w, h) * 0.10)
            im = im.crop((margin, margin, w - margin, h - margin))
            small = im.resize((downsample, downsample), Image.BILINEAR)
            arr = np.asarray(small, dtype=np.float32) / 255.0
            hist_r, _ = np.histogram(arr[..., 0], bins=16, range=(0.0, 1.0), density=True)
            hist_g, _ = np.histogram(arr[..., 1], bins=16, range=(0.0, 1.0), density=True)
            hist_b, _ = np.histogram(arr[..., 2], bins=16, range=(0.0, 1.0), density=True)
            pix = arr.reshape(-1)  # 32*32*3 = 3072 — subsample every 4th value
            pix_ds = pix[::4]
            feat = np.concatenate([hist_r, hist_g, hist_b, pix_ds]).astype(np.float32)
            features.append(feat)
            labels.append(cls_idx)

    x = np.stack(features, axis=0)
    y = np.array(labels, dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_fraction, random_state=seed, stratify=y,
    )
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    clf.fit(x_train, y_train)
    acc = float(accuracy_score(y_test, clf.predict(x_test)))
    return acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BalancedPV — medium-difficulty synthetic PlantVillage "
            "(38 class names, demo-friendly ~90–95% CNN target after training)."
        ),
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--images_per_class", type=int, default=DEFAULT_IMAGES_PER_CLASS)
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--force", action="store_true", help="Regenerate even if images exist.")
    parser.add_argument("--skip_knn", action="store_true", help="Skip post-generation kNN sanity check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = create_balanced_dataset(
        data_dir=args.data_dir,
        images_per_class=args.images_per_class,
        img_size=args.img_size,
        force=args.force,
    )
    if args.skip_knn:
        return
    acc = knn_sanity_check(root)
    print(f"[balanced] kNN hold-out accuracy (proxy separability): {acc * 100:.1f}%")
    if acc < 0.20:
        print(
            "[balanced] WARNING: kNN < 20% — dataset may be too hard for the demo target. "
            "Re-run with --force after easing generator parameters."
        )
    elif acc > 0.80:
        print(
            "[balanced] NOTE: kNN > 80% — dataset may be easier than intended (SampleTest-like)."
        )
    else:
        print("[balanced] kNN in a reasonable medium band (target proxy ~40–70%).")


if __name__ == "__main__":
    main()
