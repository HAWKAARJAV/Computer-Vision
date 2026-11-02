"""
dataset.py — Data Pipeline for ConvGeM-NeXt
=============================================
Handles:
  * Transform definition (resize + normalise, NO augmentation)
  * Loading an ImageFolder dataset from disk
  * Stratified 70 / 15 / 15 train / val / test splits
  * Wrapping splits into PyTorch DataLoaders

Expected on-disk layout
------------------------
data/
  PlantVillage/
    Apple___Apple_scab/
      image1.jpg
      image2.jpg
      ...
    Apple___Black_rot/
      ...
    (38 class folders total)
  PlantDoc/
    (27 class folders)          — flat layout, split 70/15/15
    OR train/<class>/ + test/<class>/  — official GitHub release; the
       published test set is kept and validation comes from train/
  SampleTest_PlantVillage/
    (same 38 official PlantVillage class folders, synthetic JPEGs)
  SampleTest_PlantDoc/
    (same 27 official PlantDoc class folders, synthetic JPEGs)
  RealisticPV/
    (same 38 official PlantVillage class folders; harder synthetic JPEGs)
  BalancedPV/
    (same 38 official PlantVillage class folders; medium synthetic JPEGs)
  Demo/
    (tiny 5-class synthetic set)

The folder names become the class labels automatically via ImageFolder.
"""

import copy
import json
import os
from typing import Tuple, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1.  Transforms
# ---------------------------------------------------------------------------

def get_transforms(img_size: int = 224) -> transforms.Compose:
    """Return the inference-style transform pipeline used for ALL splits.

    Note on augmentation
    --------------------
    The paper deliberately avoids any data augmentation (random crops, flips,
    colour jitter, etc.) to isolate the performance gains that come from the
    architectural changes (GeM pooling + custom head).  We therefore apply
    the *same* deterministic transform to train, validation, and test sets.

    Pipeline:
        1. Resize to ``img_size × img_size`` (224 for ConvNeXt-Base).
        2. Convert PIL image to float32 tensor in [0, 1].
        3. Normalise using ImageNet statistics (because the backbone was
           pre-trained on ImageNet).

    Args:
        img_size: Target square side length in pixels.

    Returns:
        A ``transforms.Compose`` object ready to pass to ``ImageFolder``.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet channel means
            std =[0.229, 0.224, 0.225],   # ImageNet channel stds
        ),
    ])



def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Field-robust training augmentation (proposed improvement, opt-in).

    The paper trains without augmentation. PlantDoc test images differ from
    training images in framing, lighting and background, so each op targets
    one of those shifts:
      * RandomResizedCrop  — scale / framing / partial leaves
      * flips + rotation   — leaf orientation
      * ColorJitter        — illumination and white balance
      * RandomErasing      — occlusion by other leaves, hands, stems
    Validation and test sets still use ``get_transforms`` (no augmentation).
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


# ---------------------------------------------------------------------------
# 1b. Official train/ + test/ layout (PlantDoc GitHub release)
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class SampleListDataset(Dataset):
    """Minimal ImageFolder-compatible dataset built from an explicit sample list.

    Exposes ``samples``, ``targets``, ``classes`` and ``class_to_idx`` exactly
    like ``torchvision.datasets.ImageFolder`` so the rest of the pipeline does
    not care which layout the data came from.
    """

    def __init__(self, samples, classes, transform=None) -> None:
        self.samples = list(samples)
        self.targets = [t for _, t in self.samples]
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _scan_class_dirs(root: str) -> dict:
    """Return ``{class_name: [image paths]}`` for every sub-folder of *root*."""
    out = {}
    for name in sorted(os.listdir(root)):
        cdir = os.path.join(root, name)
        if name.startswith(".") or not os.path.isdir(cdir):
            continue
        files = sorted(
            os.path.join(cdir, f) for f in os.listdir(cdir)
            if f.lower().endswith(IMG_EXTENSIONS) and not f.startswith(".")
        )
        if files:
            out[name] = files
    return out


def has_official_split(dataset_path: str) -> bool:
    """True when *dataset_path* holds ``train/<class>/`` and ``test/<class>/``."""
    return all(
        os.path.isdir(os.path.join(dataset_path, d)) and
        bool(_scan_class_dirs(os.path.join(dataset_path, d)))
        for d in ("train", "test")
    )


def build_official_split_dataset(dataset_path: str, transform):
    """Merge ``train/`` and ``test/`` into one indexable dataset.

    Only classes present in **both** folders are kept (PlantDoc ships an
    orphan ``Tomato two spotted spider mites leaf`` class in train only;
    dropping it gives the standard 27-class benchmark).

    Returns:
        ``(dataset, train_pool_idx, test_idx)``.
    """
    train_map = _scan_class_dirs(os.path.join(dataset_path, "train"))
    test_map = _scan_class_dirs(os.path.join(dataset_path, "test"))
    classes = sorted(set(train_map) & set(test_map))
    dropped = sorted(set(train_map) ^ set(test_map))
    if dropped:
        print(f"[dataset] Classes missing from train/ or test/ (dropped): {dropped}")
    samples = []
    for split_map in (train_map, test_map):
        for ci, cname in enumerate(classes):
            samples.extend((p, ci) for p in split_map[cname])
    n_train = sum(len(train_map[c]) for c in classes)
    ds = SampleListDataset(samples, classes, transform=transform)
    return ds, np.arange(n_train), np.arange(n_train, len(samples))


def open_dataset(dataset_path: str, transform):
    """Load *dataset_path* as either a flat ImageFolder or an official split.

    Returns:
        ``(dataset, official)`` where *official* is ``None`` for a flat folder
        or ``(train_pool_idx, test_idx)`` for a ``train/`` + ``test/`` layout.
    """
    if has_official_split(dataset_path):
        ds, pool_idx, test_idx = build_official_split_dataset(dataset_path, transform)
        return ds, (pool_idx, test_idx)
    return datasets.ImageFolder(root=dataset_path, transform=transform), None

# ---------------------------------------------------------------------------
# 2.  Dataset loading + stratified splitting
# ---------------------------------------------------------------------------

# Share of the official train/ folder held out for validation
# (0.15 / 0.85 keeps the same train:val ratio as the 70/15/15 split).
VAL_FRACTION_OF_POOL = 0.15 / 0.85


def split_cache_path(data_dir: str, dataset_name: str, random_seed: int) -> str:
    """Path to the reusable train/val/test index file for one dataset + seed."""
    return os.path.join(data_dir, f"{dataset_name}_split_seed{random_seed}.json")


def save_split_indices(
    path: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n_total: int,
    random_seed: int,
    dataset_name: str,
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "random_seed": random_seed,
        "n_total": int(n_total),
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
        "test_idx": [int(i) for i in test_idx],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"[dataset] Saved reusable split → {path}")


def load_split_indices(path: str, n_total: int, random_seed: int) -> Optional[tuple]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    if int(payload.get("n_total", -1)) != int(n_total):
        print(f"[dataset] Split cache ignored (sample count changed): {path}")
        return None
    if int(payload.get("random_seed", -1)) != int(random_seed):
        print(f"[dataset] Split cache ignored (seed mismatch): {path}")
        return None
    train_idx = np.array(payload["train_idx"], dtype=np.int64)
    val_idx = np.array(payload["val_idx"], dtype=np.int64)
    test_idx = np.array(payload["test_idx"], dtype=np.int64)
    print(f"[dataset] Reusing saved split → {path}")
    return train_idx, val_idx, test_idx


def get_dataloaders(
    data_dir: str,
    dataset_name: str,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    random_seed: int = 42,
    split_file: Optional[str] = None,
    augment: bool = False,
    balanced_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder]:
    """Load a dataset and return train / val / test DataLoaders.

    Splitting strategy
    ------------------
    We use *stratified* splitting so that every class is proportionally
    represented in each split — crucial for imbalanced disease datasets.

    Step 1 : 70 % train  vs  30 % temp      (stratified by class label)
    Step 2 : 50 % val   vs  50 % test        (from the 30 % temp pool)
    Result : 70 % / 15 % / 15 %             (of the original data)

    Args:
        data_dir:     Root data directory (e.g. ``"data"``).
        dataset_name: Sub-folder name (``"PlantVillage"`` or ``"PlantDoc"``).
        batch_size:   Samples per mini-batch.
        img_size:     Image side length passed to ``get_transforms()``.
        num_workers:  CPU workers for parallel data loading.
        pin_memory:   Whether to pin host memory for faster GPU transfer.
        random_seed:  Seed for reproducible splits.
        augment:      Apply ``get_train_transforms`` to the train split only.
        balanced_sampler: Sample training images inversely to class frequency
                      (helps minority PlantDoc classes; raises MCA).

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader, full_dataset)``.
        ``full_dataset.classes`` gives the list of class names.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
    """
    try:
        from config import resolve_dataset_name
        dataset_name = resolve_dataset_name(dataset_name)
    except Exception:
        pass

    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: '{dataset_path}'\n"
            f"Expected structure: {dataset_path}/<class_name>/*.jpg"
        )

    # ---- Load full dataset with ImageFolder --------------------------------
    transform = get_transforms(img_size=img_size)
    full_dataset, official = open_dataset(dataset_path, transform)

    num_classes = len(full_dataset.classes)
    n_total = len(full_dataset)

    print(f"\n{'='*55}")
    print(f"Dataset       : {dataset_name}")
    print(f"Path          : {dataset_path}")
    print(f"Classes       : {num_classes}")
    print(f"Total samples : {n_total:,}")
    print(f"{'='*55}")

    # ---- Extract per-sample integer labels for stratification --------------
    # full_dataset.targets is a list of int labels matching full_dataset.samples
    labels: List[int] = full_dataset.targets

    cache_path = split_file or split_cache_path(data_dir, dataset_name, random_seed)
    cached = load_split_indices(cache_path, n_total, random_seed)
    if cached is not None:
        train_idx, val_idx, test_idx = cached
    elif official is not None:
        # ---- Official train/ + test/ layout: keep the published test set,
        #      carve a stratified validation set out of the train pool -------
        pool_idx, test_idx = official
        val_frac = VAL_FRACTION_OF_POOL
        pool_labels = [labels[i] for i in pool_idx]
        train_idx, val_idx = train_test_split(
            pool_idx,
            test_size=val_frac,
            stratify=pool_labels,
            random_state=random_seed,
        )
        print(f"[dataset] Official split detected: train/ → train+val "
              f"({100*(1-val_frac):.0f}/{100*val_frac:.0f}), test/ → test")
        save_split_indices(
            cache_path, train_idx, val_idx, test_idx,
            n_total=n_total, random_seed=random_seed, dataset_name=dataset_name,
        )
    else:
        # ---- Stratified split 1: train (70%) vs temp (30%) ----------------
        all_indices = np.arange(n_total)
        train_idx, temp_idx = train_test_split(
            all_indices,
            test_size=0.30,
            stratify=labels,
            random_state=random_seed,
        )

        # ---- Stratified split 2: val (15%) vs test (15%) from temp --------
        temp_labels = [labels[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,        # 50% of 30% = 15% overall
            stratify=temp_labels,
            random_state=random_seed,
        )
        save_split_indices(
            cache_path, train_idx, val_idx, test_idx,
            n_total=n_total, random_seed=random_seed, dataset_name=dataset_name,
        )

    print(f"Train samples : {len(train_idx):,}  ({100*len(train_idx)/n_total:.1f}%)")
    print(f"Val   samples : {len(val_idx):,}   ({100*len(val_idx)/n_total:.1f}%)")
    print(f"Test  samples : {len(test_idx):,}   ({100*len(test_idx)/n_total:.1f}%)")
    print(f"{'='*55}\n")

    # ---- Wrap index lists into Subset objects ------------------------------
    train_source = full_dataset
    if augment:
        # Shallow copy shares the sample list but carries its own transform,
        # so val/test keep the deterministic pipeline.
        train_source = copy.copy(full_dataset)
        train_source.transform = get_train_transforms(img_size=img_size)
        print("[dataset] Train-time augmentation: ON")
    train_subset = Subset(train_source, train_idx)
    val_subset   = Subset(full_dataset, val_idx)
    test_subset  = Subset(full_dataset, test_idx)

    # ---- Build DataLoaders -------------------------------------------------
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    sampler = None
    if balanced_sampler:
        train_labels = np.array([labels[i] for i in train_idx])
        counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        weights = 1.0 / counts.clip(min=1)[train_labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_idx),
            replacement=True,
            generator=torch.Generator().manual_seed(random_seed),
        )
        print("[dataset] Class-balanced sampler: ON")

    train_loader = DataLoader(
        train_subset,
        shuffle=sampler is None,   # Shuffle training data each epoch
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        shuffle=False,         # Keep val/test order fixed for reproducibility
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_subset,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, full_dataset


def load_imagefolder(
    data_dir: str,
    dataset_name: str,
    img_size: int = 224,
) -> datasets.ImageFolder:
    """Load a full ImageFolder dataset without splitting."""
    try:
        from config import resolve_dataset_name
        dataset_name = resolve_dataset_name(dataset_name)
    except Exception:
        pass
    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: '{dataset_path}'\n"
            f"Expected structure: {dataset_path}/<class_name>/*.jpg"
        )
    transform = get_transforms(img_size=img_size)
    return open_dataset(dataset_path, transform)[0]


def make_loader(
    dataset: datasets.ImageFolder,
    indices: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader from a subset of dataset indices."""
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import config as cfg

    # Quick sanity check — requires the dataset to already be on disk.
    data_dir = cfg.DATA_DIR
    dataset_name = cfg.DATASET_NAME

    try:
        train_loader, val_loader, test_loader, dataset = get_dataloaders(
            data_dir=data_dir,
            dataset_name=dataset_name,
            batch_size=cfg.BATCH_SIZE,
            img_size=cfg.IMG_SIZE,
            num_workers=0,              # 0 workers is safer in __main__ on some OSes
            pin_memory=False,
            random_seed=cfg.RANDOM_SEED,
        )
    except FileNotFoundError as e:
        print(f"[dataset.py smoke-test] {e}")
        print("Please download the dataset and place it under the 'data/' folder.")
        sys.exit(0)

    # Grab one batch and inspect shapes
    images, labels = next(iter(train_loader))
    print(f"Batch image shape : {tuple(images.shape)}")
    print(f"Batch label shape : {tuple(labels.shape)}")
    print(f"Class names (first 5): {dataset.classes[:5]}")
