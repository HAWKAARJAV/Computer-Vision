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
    (27 class folders)

The folder names become the class labels automatically via ImageFolder.
"""

import os
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
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


# ---------------------------------------------------------------------------
# 2.  Dataset loading + stratified splitting
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_dir: str,
    dataset_name: str,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    random_seed: int = 42,
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

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader, full_dataset)``.
        ``full_dataset.classes`` gives the list of class names.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
    """
    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: '{dataset_path}'\n"
            f"Expected structure: {dataset_path}/<class_name>/*.jpg"
        )

    # ---- Load full dataset with ImageFolder --------------------------------
    transform = get_transforms(img_size=img_size)
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

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

    # ---- Stratified split 1: train (70%) vs temp (30%) --------------------
    all_indices = np.arange(n_total)
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.30,
        stratify=labels,
        random_state=random_seed,
    )

    # ---- Stratified split 2: val (15%) vs test (15%) from temp (30%) ------
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,        # 50% of 30% = 15% overall
        stratify=temp_labels,
        random_state=random_seed,
    )

    print(f"Train samples : {len(train_idx):,}  ({100*len(train_idx)/n_total:.1f}%)")
    print(f"Val   samples : {len(val_idx):,}   ({100*len(val_idx)/n_total:.1f}%)")
    print(f"Test  samples : {len(test_idx):,}   ({100*len(test_idx)/n_total:.1f}%)")
    print(f"{'='*55}\n")

    # ---- Wrap index lists into Subset objects ------------------------------
    train_subset = Subset(full_dataset, train_idx)
    val_subset   = Subset(full_dataset, val_idx)
    test_subset  = Subset(full_dataset, test_idx)

    # ---- Build DataLoaders -------------------------------------------------
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(
        train_subset,
        shuffle=True,          # Shuffle training data each epoch
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
    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: '{dataset_path}'\n"
            f"Expected structure: {dataset_path}/<class_name>/*.jpg"
        )
    transform = get_transforms(img_size=img_size)
    return datasets.ImageFolder(root=dataset_path, transform=transform)


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
