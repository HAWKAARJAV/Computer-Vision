"""
create_realistic_dataset.py — Harder synthetic PlantVillage-style set.

Every class uses the same ellipse leaf, vein layout, and soil background.
Class identity is a subtle lesion cue whose parameters overlap related
diseases (early blight vs late blight vs target spot, bacterial spots,
mildews, rusts, yellowing viruses). Per-image lighting, blur, shadow,
clutter, and pose vary widely. Images are synthetic — not PlantVillage
photos — and carry no painted class name and no label noise.

    data/RealisticPV/<38 official PlantVillage class folders>/*.jpg

Usage:
    python create_realistic_dataset.py
    python create_realistic_dataset.py --images_per_class 40 --force
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from create_sample_test_dataset import PLANTVILLAGE_CLASSES


CANONICAL_NAME = "RealisticPV"
DEFAULT_IMAGES_PER_CLASS = 40
DEFAULT_IMG_SIZE = 224

# Weak crop tint only. Offsets are a few degrees inside a shared green
# band and are smaller than per-image hue noise, so colour is not a
# reliable class id (unlike SampleTest, which spreads hues around the wheel).
_CROP_HUE_BIAS: Dict[str, float] = {
    "apple": 0.0,
    "blueberry": 5.0,
    "cherry": -3.0,
    "corn": -7.0,
    "grape": 6.0,
    "orange": -5.0,
    "peach": 2.0,
    "pepper": 4.0,
    "potato": -4.0,
    "raspberry": 3.0,
    "soybean": -6.0,
    "squash": -1.0,
    "strawberry": 7.0,
    "tomato": 1.0,
}
_CROP_VAL_BIAS: Dict[str, float] = {
    "apple": 0.00,
    "blueberry": -0.04,
    "cherry": 0.02,
    "corn": 0.03,
    "grape": -0.02,
    "orange": 0.02,
    "peach": 0.01,
    "pepper": 0.03,
    "potato": -0.03,
    "raspberry": 0.00,
    "soybean": 0.02,
    "squash": 0.04,
    "strawberry": -0.01,
    "tomato": 0.00,
}

# Lesion recipes. Ranges are intentionally wide and shared inside a family
# so a single image often looks legal for a neighbouring class.
# Keys: n, r (px), color (RGB 0-1), alpha, elongate, ring (halo probability),
# edge (margin bias 0-1), hue_shift (degrees added to leaf hue; negative
# moves green toward yellow), kind, light_center.
_Symptom = Dict[str, object]


def _sym(
    n: Tuple[int, int],
    r: Tuple[float, float],
    color: Tuple[float, float, float],
    alpha: float,
    elongate: float = 1.15,
    ring: float = 0.0,
    edge: float = 0.0,
    hue_shift: Tuple[float, float] = (0.0, 0.0),
    kind: str = "spots",
    light_center: bool = False,
) -> _Symptom:
    return {
        "n": n,
        "r": r,
        "color": color,
        "alpha": alpha,
        "elongate": elongate,
        "ring": ring,
        "edge": edge,
        "hue_shift": hue_shift,
        "kind": kind,
        "light_center": light_center,
    }


SYMPTOMS: Dict[str, _Symptom] = {
    # Olive-brown scab vs larger black rot vs orange rust — rust overlaps corn rust.
    "Apple___Apple_scab": _sym(
        (6, 14), (2.0, 4.4), (0.40, 0.36, 0.18), 0.46, elongate=1.3, ring=0.2, edge=0.2,
    ),
    "Apple___Black_rot": _sym(
        (2, 7), (4.0, 9.0), (0.18, 0.12, 0.09), 0.58, elongate=1.2, ring=0.25,
    ),
    "Apple___Cedar_apple_rust": _sym(
        (5, 13), (2.0, 4.6), (0.72, 0.42, 0.14), 0.52, elongate=1.05, ring=0.35, kind="rust",
    ),
    "Cherry_(including_sour)___Powdery_mildew": _sym(
        (12, 26), (2.0, 4.6), (0.82, 0.82, 0.78), 0.32, elongate=1.1, kind="mildew",
    ),
    # Gray elongated spots vs orange pustules vs long tan streaks.
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": _sym(
        (4, 9), (2.4, 4.5), (0.46, 0.42, 0.32), 0.48, elongate=2.8, ring=0.1,
    ),
    "Corn_(maize)___Common_rust_": _sym(
        (12, 26), (1.4, 3.2), (0.70, 0.40, 0.13), 0.50, elongate=1.1, ring=0.2, kind="rust",
    ),
    "Corn_(maize)___Northern_Leaf_Blight": _sym(
        (2, 5), (3.0, 5.5), (0.55, 0.40, 0.20), 0.50, elongate=4.4, kind="streak",
    ),
    "Grape___Black_rot": _sym(
        (4, 11), (2.2, 5.5), (0.20, 0.13, 0.10), 0.55, elongate=1.05, ring=0.65,
    ),
    "Grape___Esca_(Black_Measles)": _sym(
        (3, 7), (6.0, 12.0), (0.70, 0.64, 0.28), 0.36, elongate=2.2,
        hue_shift=(-14.0, -5.0), kind="mosaic",
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": _sym(
        (6, 13), (3.0, 6.5), (0.46, 0.28, 0.14), 0.48, elongate=1.25, ring=0.25,
    ),
    "Orange___Haunglongbing_(Citrus_greening)": _sym(
        (3, 6), (8.0, 16.0), (0.74, 0.68, 0.26), 0.40, elongate=1.4,
        hue_shift=(-18.0, -8.0), kind="mosaic",
    ),
    # Bacterial spots: nearly the same tiny dark lesions, slight count/colour shifts.
    "Peach___Bacterial_spot": _sym(
        (10, 22), (1.2, 2.6), (0.32, 0.14, 0.10), 0.55, elongate=1.05,
    ),
    "Pepper,_bell___Bacterial_spot": _sym(
        (9, 21), (1.3, 2.8), (0.38, 0.16, 0.11), 0.52, elongate=1.08,
    ),
    # Early / late / target: shared brown spots, small mean shifts in size and halo rate.
    "Potato___Early_blight": _sym(
        (4, 10), (3.4, 7.2), (0.48, 0.31, 0.15), 0.50, elongate=1.15, ring=0.40,
    ),
    "Potato___Late_blight": _sym(
        (3, 8), (5.2, 10.5), (0.32, 0.18, 0.10), 0.52, elongate=1.25, ring=0.12,
    ),
    "Squash___Powdery_mildew": _sym(
        (14, 30), (2.2, 5.0), (0.84, 0.84, 0.80), 0.36, elongate=1.15, kind="mildew",
    ),
    "Strawberry___Leaf_scorch": _sym(
        (7, 15), (2.6, 5.5), (0.44, 0.24, 0.12), 0.50, elongate=1.3, ring=0.15, edge=0.85,
    ),
    "Tomato___Bacterial_spot": _sym(
        (12, 24), (1.1, 2.5), (0.30, 0.12, 0.09), 0.56, elongate=1.02,
    ),
    "Tomato___Early_blight": _sym(
        (5, 11), (3.6, 7.6), (0.50, 0.32, 0.16), 0.50, elongate=1.12, ring=0.50,
    ),
    "Tomato___Late_blight": _sym(
        (3, 9), (5.0, 10.2), (0.34, 0.20, 0.11), 0.52, elongate=1.3, ring=0.18,
    ),
    "Tomato___Leaf_Mold": _sym(
        (4, 9), (5.5, 11.0), (0.68, 0.62, 0.30), 0.38, elongate=1.35,
        hue_shift=(-10.0, -3.0), kind="mildew",
    ),
    "Tomato___Septoria_leaf_spot": _sym(
        (10, 20), (1.8, 3.6), (0.40, 0.24, 0.12), 0.52, elongate=1.05, ring=0.75,
        light_center=True,
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": _sym(
        (28, 55), (0.7, 1.5), (0.76, 0.72, 0.46), 0.38, elongate=1.0, kind="stipple",
    ),
    "Tomato___Target_Spot": _sym(
        (6, 13), (2.6, 6.2), (0.50, 0.33, 0.16), 0.50, elongate=1.05, ring=0.82,
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": _sym(
        (3, 7), (6.0, 12.0), (0.74, 0.70, 0.30), 0.32, elongate=1.5,
        hue_shift=(-16.0, -6.0), kind="mosaic",
    ),
    "Tomato___Tomato_mosaic_virus": _sym(
        (5, 11), (5.0, 11.0), (0.62, 0.66, 0.28), 0.34, elongate=1.3,
        hue_shift=(-6.0, 2.0), kind="mosaic",
    ),
}

_HEALTHY_DUST: _Symptom = _sym(
    (0, 3), (1.0, 2.2), (0.36, 0.28, 0.16), 0.22, elongate=1.1, kind="spots",
)

_GRID_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _assert_catalog() -> None:
    missing = [
        name for name in PLANTVILLAGE_CLASSES
        if "healthy" not in name.lower() and name not in SYMPTOMS
    ]
    extra = [name for name in SYMPTOMS if name not in PLANTVILLAGE_CLASSES]
    if missing or extra or len(PLANTVILLAGE_CLASSES) != 38:
        raise RuntimeError(f"Symptom catalog mismatch. missing={missing} extra={extra}")


_assert_catalog()


def _crop_key(class_name: str) -> str:
    head = class_name.split("___", 1)[0]
    head = head.split("(")[0].split(",")[0]
    return head.strip().lower()


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    h = h % 360.0
    s = float(np.clip(s, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0.0
    elif h < 120:
        r, g, b = x, c, 0.0
    elif h < 180:
        r, g, b = 0.0, c, x
    elif h < 240:
        r, g, b = 0.0, x, c
    elif h < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return np.array([r + m, g + m, b + m], dtype=np.float32)


def _xy(size: int) -> Tuple[np.ndarray, np.ndarray]:
    cached = _GRID_CACHE.get(size)
    if cached is None:
        yy, xx = np.mgrid[0:size, 0:size]
        cached = (yy.astype(np.float32), xx.astype(np.float32))
        _GRID_CACHE[size] = cached
    return cached


def _blend_gaussian(
    img: np.ndarray,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    ang: float,
    color: np.ndarray,
    alpha: float,
    mask: Optional[np.ndarray] = None,
) -> None:
    size = img.shape[0]
    rad = int(math.ceil(3.0 * max(sx, sy))) + 2
    x0 = max(0, int(math.floor(cx)) - rad)
    x1 = min(size, int(math.floor(cx)) + rad + 1)
    y0 = max(0, int(math.floor(cy)) - rad)
    y1 = min(size, int(math.floor(cy)) + rad + 1)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx.astype(np.float32) - np.float32(cx)
    dy = yy.astype(np.float32) - np.float32(cy)
    ca = math.cos(ang)
    sa = math.sin(ang)
    xr = ca * dx + sa * dy
    yr = -sa * dx + ca * dy
    blob = np.exp(-0.5 * ((xr / max(sx, 0.4)) ** 2 + (yr / max(sy, 0.4)) ** 2)).astype(np.float32)
    if mask is not None:
        blob *= mask[y0:y1, x0:x1]
    w = (np.float32(alpha) * blob)[..., None]
    region = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = region * (1.0 - w) + color.reshape(1, 1, 3) * w


def _sample_leaf_color(rng: np.random.Generator, class_name: str, symptom: _Symptom) -> np.ndarray:
    crop = _crop_key(class_name)
    hue_bias = _CROP_HUE_BIAS.get(crop, 0.0)
    val_bias = _CROP_VAL_BIAS.get(crop, 0.0)
    shift_lo, shift_hi = symptom["hue_shift"]  # type: ignore[misc]
    hue = float(rng.normal(100.0 + hue_bias, 8.0) + rng.uniform(shift_lo, shift_hi))
    sat = float(np.clip(rng.normal(0.46, 0.09), 0.26, 0.68))
    val = float(np.clip(rng.normal(0.50 + val_bias, 0.08), 0.30, 0.74))
    return _hsv_to_rgb(hue, sat, val)


def _paint_symptom(
    img: np.ndarray,
    mask: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    leaf_ang: float,
    symptom: _Symptom,
    rng: np.random.Generator,
) -> None:
    n_lo, n_hi = symptom["n"]  # type: ignore[misc]
    n = int(rng.integers(n_lo, n_hi + 1))
    kind = symptom["kind"]
    r_lo, r_hi = symptom["r"]  # type: ignore[misc]
    base_color = np.array(symptom["color"], dtype=np.float32)
    edge = float(symptom["edge"])
    elongate = float(symptom["elongate"])
    ring_p = float(symptom["ring"])
    light_center = bool(symptom["light_center"])
    ca = math.cos(leaf_ang)
    sa = math.sin(leaf_ang)

    for _ in range(n):
        if edge > 0 and rng.random() < (0.35 + 0.55 * edge):
            rad = float(rng.uniform(0.58, 0.92))
        else:
            rad = math.sqrt(float(rng.random())) * 0.86
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        lx = rad * rx * math.cos(theta)
        ly = rad * ry * math.sin(theta)
        x = cx + ca * lx - sa * ly
        y = cy + sa * lx + ca * ly
        radius = float(rng.uniform(r_lo, r_hi))
        spin = leaf_ang + float(rng.uniform(-0.5, 0.5))
        aspect = elongate * float(rng.uniform(0.75, 1.25))
        if kind == "streak":
            sx = radius * float(rng.uniform(0.55, 0.9))
            sy = radius * aspect
        else:
            sx = radius * float(rng.uniform(0.75, 1.15))
            sy = radius * aspect * float(rng.uniform(0.8, 1.2))
        color = np.clip(base_color + rng.normal(0.0, 0.06, 3).astype(np.float32), 0.0, 1.0)
        alpha = float(np.clip(float(symptom["alpha"]) * rng.uniform(0.6, 1.15), 0.1, 0.8))

        if kind == "mosaic":
            if rng.random() < 0.5:
                color = np.clip(color + np.array([0.08, 0.06, -0.02], dtype=np.float32), 0, 1)
            else:
                color = np.clip(color * 0.72, 0, 1)
            alpha *= 0.85
        elif kind == "stipple":
            alpha *= float(rng.uniform(0.7, 1.1))

        if ring_p > 0 and rng.random() < ring_p:
            halo = np.clip(color * 1.35 + 0.08, 0, 1)
            _blend_gaussian(img, x, y, sx * 1.7, sy * 1.7, spin, halo, alpha * 0.45, mask)
        _blend_gaussian(img, x, y, sx, sy, spin, color, alpha, mask)
        if light_center and rng.random() < 0.7:
            center = np.clip(color * 1.5 + 0.15, 0, 1)
            _blend_gaussian(img, x, y, sx * 0.45, sy * 0.45, spin, center, alpha * 0.7, mask)
        elif kind == "rust" and rng.random() < 0.55:
            core = np.clip(color * 0.55, 0, 1)
            _blend_gaussian(img, x, y, sx * 0.4, sy * 0.4, spin, core, alpha * 0.6, mask)


def make_leaf_image(class_name: str, seed: int, size: int = DEFAULT_IMG_SIZE) -> Image.Image:
    """One synthetic leaf. Same geometry family for every class."""
    rng = np.random.default_rng(seed)
    symptom = SYMPTOMS.get(class_name, _HEALTHY_DUST)
    yy, xx = _xy(size)

    soil = np.array([0.40, 0.30, 0.20], dtype=np.float32)
    soil = np.clip(soil * np.float32(rng.uniform(0.82, 1.18)) + rng.normal(0, 0.015, 3), 0, 1)
    img = np.broadcast_to(soil, (size, size, 3)).astype(np.float32).copy()

    coarse = rng.normal(0, 1, (14, 14)).astype(np.float32)
    coarse_img = Image.fromarray(np.clip(coarse * 28 + 128, 0, 255).astype(np.uint8), mode="L")
    coarse_big = np.asarray(coarse_img.resize((size, size), Image.BILINEAR), dtype=np.float32)
    img += ((coarse_big - 128.0) / 128.0 * 0.055)[..., None]
    img += rng.normal(0, 0.012, (size, size, 1)).astype(np.float32)

    # Pebbles in the soil, same distribution for every class.
    for _ in range(int(rng.integers(7, 16))):
        px = float(rng.uniform(0, size - 1))
        py = float(rng.uniform(0, size - 1))
        pr = float(rng.uniform(1.4, 5.2))
        pebble = np.clip(soil * np.float32(rng.uniform(0.65, 1.35)), 0, 1)
        _blend_gaussian(
            img, px, py, pr, pr * float(rng.uniform(0.55, 1.35)),
            float(rng.uniform(0, math.pi)), pebble, float(rng.uniform(0.35, 0.7)),
        )

    ang = float(rng.uniform(-14.0, 14.0) * math.pi / 180.0)
    scale = float(rng.uniform(0.94, 1.06))
    rx = size * 0.23 * scale
    ry = size * 0.36 * scale
    cx = size * 0.5 + float(rng.uniform(-16, 16))
    cy = size * 0.5 + float(rng.uniform(-14, 14))
    ca = math.cos(ang)
    sa = math.sin(ang)
    dx = xx - np.float32(cx)
    dy = yy - np.float32(cy)
    xr = ca * dx + sa * dy
    yr = -sa * dx + ca * dy
    rnorm = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    mask = np.clip((1.02 - rnorm) / 0.035, 0.0, 1.0).astype(np.float32)

    leaf_color = _sample_leaf_color(rng, class_name, symptom)
    mottling_small = rng.normal(0, 1, (12, 12, 3)).astype(np.float32)
    mottling_u8 = np.clip(mottling_small * 32 + 128, 0, 255).astype(np.uint8)
    mottling = np.asarray(
        Image.fromarray(mottling_u8, mode="RGB").resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    )
    mottling = (mottling - 128.0) / 128.0 * 0.045
    leaf_rgb = np.clip(leaf_color.reshape(1, 1, 3) + mottling, 0, 1)
    img = img * (1.0 - mask[..., None]) + leaf_rgb * mask[..., None]

    # Shared vein template: one midrib and four lateral pairs, tiny jitter only.
    vein = np.exp(-0.5 * (xr / 1.15) ** 2) * (np.abs(yr) < ry * 0.92)
    for side in (-1.0, 1.0):
        for t, spread in ((-0.55, 0.22), (-0.22, 0.16), (0.12, 0.18), (0.42, 0.20)):
            y0 = (t + float(rng.uniform(-0.03, 0.03))) * ry
            x1 = side * rx * 0.78
            y1 = y0 + side * float(rng.uniform(-spread, spread)) * ry * 0.15 + spread * ry * 0.05
            vx, vy = x1 - 0.0, y1 - y0
            length2 = vx * vx + vy * vy + 1e-6
            proj = np.clip((xr * vx + (yr - y0) * vy) / length2, 0.0, 1.0)
            dist = np.hypot(xr - proj * vx, (yr - y0) - proj * vy)
            vein = np.maximum(vein, np.exp(-0.5 * (dist / 0.85) ** 2) * (proj > 0) * (proj < 1))
    vein *= mask
    darken = (0.10 + float(rng.uniform(-0.02, 0.02))) * vein[..., None]
    img = np.clip(img - darken, 0, 1)

    _paint_symptom(img, mask, cx, cy, rx, ry, ang, symptom, rng)

    # Occasional soil clod occluding the leaf margin.
    if rng.random() < 0.30:
        for _ in range(int(rng.integers(1, 3))):
            theta = float(rng.uniform(0, 2 * math.pi))
            ox = cx + math.cos(theta) * rx * float(rng.uniform(0.75, 1.15))
            oy = cy + math.sin(theta) * ry * float(rng.uniform(0.75, 1.15))
            clod = np.clip(soil * np.float32(rng.uniform(0.75, 1.15)), 0, 1)
            _blend_gaussian(
                img, ox, oy,
                float(rng.uniform(6, 14)), float(rng.uniform(4, 11)),
                float(rng.uniform(0, math.pi)), clod, float(rng.uniform(0.45, 0.75)),
            )

    if rng.random() < 0.68:
        shade_ang = float(rng.uniform(0, 2 * math.pi))
        ramp = xx * math.cos(shade_ang) + yy * math.sin(shade_ang)
        ramp = (ramp - ramp.min()) / (float(ramp.max() - ramp.min()) + 1e-6)
        cut = float(rng.uniform(0.25, 0.62))
        soft = np.clip((ramp - cut) / 0.28, 0.0, 1.0).astype(np.float32)
        darkness = float(rng.uniform(0.55, 0.82))
        img *= (1.0 - soft[..., None] * (1.0 - darkness))

    gain = float(rng.uniform(0.72, 1.30))
    contrast = float(rng.uniform(0.78, 1.24))
    mid = float(np.median(img))
    img = (img - mid) * contrast + mid * gain
    img = np.clip(img, 0.0, 1.0)
    gamma = float(rng.uniform(0.88, 1.16))
    img = np.power(img, gamma)

    u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB")
    if rng.random() < 0.40:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.45, 1.35))))
    return pil


def _count_jpegs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.jpg"))


def create_realistic_dataset(
    data_dir: str = "data",
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
    img_size: int = DEFAULT_IMG_SIZE,
    force: bool = False,
    dataset_name: str = CANONICAL_NAME,
) -> str:
    """Write ``data/RealisticPV``. Returns the dataset root path."""
    if images_per_class < 1:
        raise ValueError("images_per_class must be >= 1")
    if dataset_name != CANONICAL_NAME:
        raise ValueError(f"This generator only writes {CANONICAL_NAME}.")

    root = Path(data_dir) / dataset_name
    if root.exists() and not force:
        existing = _count_jpegs(root)
        if existing > 0:
            print(f"[realistic] Already exists: {root} ({existing} images)")
            print("[realistic] Use --force to regenerate.")
            return str(root)

    if force and root.exists():
        shutil.rmtree(root)

    print(f"[realistic] Creating {dataset_name} at {root}/")
    print(
        "[realistic] Shared leaf template; overlapping lesion cues; "
        f"{len(PLANTVILLAGE_CLASSES)} classes × {images_per_class} images."
    )
    for cls_idx, class_name in enumerate(PLANTVILLAGE_CLASSES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = make_leaf_image(
                class_name=class_name,
                seed=20_260 + cls_idx * 10_000 + i,
                size=img_size,
            )
            img.save(class_dir / f"{i:03d}.jpg", quality=82, subsampling=2)
        print(f"[realistic]   {cls_idx + 1:02d}/38  {class_name}")

    total = images_per_class * len(PLANTVILLAGE_CLASSES)
    print(f"[realistic] Done — {len(PLANTVILLAGE_CLASSES)} classes × {images_per_class} = {total} → {root}")
    return str(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the synthetic RealisticPV dataset "
            "(38 PlantVillage class names, shared leaf, subtle overlapping lesions)."
        ),
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--images_per_class", type=int, default=DEFAULT_IMAGES_PER_CLASS)
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if images already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_realistic_dataset(
        data_dir=args.data_dir,
        images_per_class=args.images_per_class,
        img_size=args.img_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
