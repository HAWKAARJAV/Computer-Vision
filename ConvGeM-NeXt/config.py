"""
config.py — Central Configuration for ConvGeM-NeXt
====================================================
All hyperparameters, dataset settings, and paths live here.
Import this module anywhere in the project to access settings.
"""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"                   # Root directory that contains dataset sub-folders
CHECKPOINT_DIR = "checkpoints"      # Where best_model.pth will be saved
LOGS_DIR = "logs"                   # CSV training logs and plot PNGs are saved here


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "PlantVillage"       # Default dataset; override via CLI or direct assignment

# Number of output classes per dataset
NUM_CLASSES_PLANTVILL = 38          # PlantVillage: 38 disease + healthy classes
NUM_CLASSES_PLANTDOC = 27           # PlantDoc: 27 classes (real-world, harder benchmark)
NUM_CLASSES_DEMO = 5                # Tiny synthetic demo set (no download needed)


# ---------------------------------------------------------------------------
# Data Split Ratios  (must sum to 1.0)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70     # 70% of data used for training
VAL_RATIO   = 0.15     # 15% used for validation (hyperparameter tuning / early stopping)
TEST_RATIO  = 0.15     # 15% held out for final evaluation — never seen during training

RANDOM_SEED = 42       # Fixed seed for reproducible splits across runs


# ---------------------------------------------------------------------------
# Image Settings
# ---------------------------------------------------------------------------
IMG_SIZE = 224          # ConvNeXt-Base expects 224×224 input (same as ImageNet pre-training)

# ImageNet normalisation constants — used because backbone was pre-trained on ImageNet
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE    = 64       # Fits comfortably on a single 16 GB GPU; lower if OOM
NUM_EPOCHS    = 30       # Paper trains for 30 epochs
LEARNING_RATE = 1e-3     # AdamW initial learning rate
WEIGHT_DECAY  = 1e-4     # L2 regularisation coefficient in AdamW

# CosineAnnealingLR: LR decays from LEARNING_RATE → ~0 over T_MAX epochs
T_MAX = 30               # Set equal to NUM_EPOCHS for a single full cosine cycle


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
DROPOUT    = 0.3        # Dropout probability in the custom classifier head
GEM_P_INIT = 3.0        # Initial value of the learnable GeM exponent `p`
                         # p=1 → average pooling; p→∞ → max pooling


# ---------------------------------------------------------------------------
# DataLoader Performance
# ---------------------------------------------------------------------------
NUM_WORKERS = 4          # Parallel CPU worker processes for data loading
                         # Set to 0 on Windows or if you encounter issues
PIN_MEMORY  = True       # Pins host memory for faster host→GPU transfer
                         # Set to False when running on CPU only


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_num_classes(dataset_name: str) -> int:
    """Return the number of output classes for the specified dataset.

    Args:
        dataset_name: One of ``'PlantVillage'`` or ``'PlantDoc'`` (case-sensitive).

    Returns:
        Integer number of classes.

    Raises:
        ValueError: If *dataset_name* is not recognised.
    """
    mapping = {
        "PlantVillage": NUM_CLASSES_PLANTVILL,
        "PlantDoc":     NUM_CLASSES_PLANTDOC,
        "Demo":         NUM_CLASSES_DEMO,
    }
    if dataset_name not in mapping:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported datasets: {list(mapping.keys())}"
        )
    return mapping[dataset_name]
