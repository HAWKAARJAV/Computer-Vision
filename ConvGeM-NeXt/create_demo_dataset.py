"""
create_demo_dataset.py — Tiny synthetic plant-disease dataset (no download).

Generates ~100 small JPEG images in 5 class folders under data/Demo/.
Each class uses a distinct colour + texture so a quick training run can learn.

Usage:
    python create_demo_dataset.py
    python create_demo_dataset.py --images_per_class 20
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


DEMO_CLASSES = [
    "Tomato___Healthy",
    "Tomato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Corn___Common_Rust",
]

# Base RGB colours per class (leaf-like tones)
CLASS_COLORS = [
    (46, 125, 50),    # green — healthy tomato
    (139, 69, 19),    # brown — blight
    (76, 175, 80),    # light green — healthy potato
    (121, 85, 72),    # dark brown — late blight
    (255, 193, 7),    # yellow/rust — corn rust
]


def make_leaf_image(
    base_color: tuple,
    seed: int,
    size: int = 160,
) -> Image.Image:
    """Create a simple synthetic leaf image with class-specific colour/texture."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size, 3), dtype=np.uint8)

    # Fill background (soil / dark green)
    arr[:, :] = np.array([34, 50, 28], dtype=np.uint8)

    # Draw an elliptical "leaf" region
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2 + int(rng.integers(-8, 9)), size // 2 + int(rng.integers(-8, 9))
    rx, ry = int(size * 0.35), int(size * 0.22)
    color = tuple(int(c + rng.integers(-25, 26)) for c in base_color)
    draw.ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry],
        fill=color,
        outline=tuple(max(0, c - 40) for c in color),
        width=2,
    )

    # Add vein lines
    for _ in range(3):
        x0 = cx + int(rng.integers(-rx // 2, rx // 2))
        draw.line([(x0, cy - ry), (cx, cy + ry)], fill=tuple(max(0, c - 30) for c in color), width=1)

    # Class-specific spots / rust patches
    n_spots = int(rng.integers(4, 12))
    for _ in range(n_spots):
        sx = int(rng.integers(cx - rx, cx + rx))
        sy = int(rng.integers(cy - ry, cy + ry))
        spot_r = int(rng.integers(2, 7))
        spot_color = tuple(int(rng.integers(0, 256)) for _ in range(3))
        draw.ellipse([sx - spot_r, sy - spot_r, sx + spot_r, sy + spot_r], fill=spot_color)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img


def create_demo_dataset(
    data_dir: str = "data",
    dataset_name: str = "Demo",
    images_per_class: int = 20,
    img_size: int = 160,
    force: bool = False,
) -> str:
    """Write synthetic demo images to disk. Returns the dataset root path."""
    root = Path(data_dir) / dataset_name
    if root.exists() and not force:
        existing = sum(1 for _ in root.rglob("*.jpg"))
        if existing > 0:
            print(f"[demo-data] Already exists: {root} ({existing} images)")
            print("[demo-data] Use --force to regenerate.")
            return str(root)

    print(f"[demo-data] Creating synthetic dataset at {root}/")
    for cls_idx, class_name in enumerate(DEMO_CLASSES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        base_color = CLASS_COLORS[cls_idx]

        for i in range(images_per_class):
            img = make_leaf_image(base_color, seed=cls_idx * 1000 + i, size=img_size)
            img.save(class_dir / f"{class_name.lower()}_{i:03d}.jpg", quality=85)

    total = images_per_class * len(DEMO_CLASSES)
    print(f"[demo-data] Done — {len(DEMO_CLASSES)} classes × {images_per_class} images = {total} total")
    return str(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny demo plant-disease dataset.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="Demo")
    parser.add_argument("--images_per_class", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--force", action="store_true", help="Overwrite existing demo data.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_demo_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        images_per_class=args.images_per_class,
        img_size=args.img_size,
        force=args.force,
    )
