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
CHECKPOINT_DIR = "checkpoints"      # Legacy default; train.py now prefers results/<variant>/
LOGS_DIR = "logs"                   # Legacy default for older scripts / demo
RESULTS_DIR = "results"             # Ablation outputs: results/<variant>/


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "PlantVillage"       # Default dataset; override via CLI or direct assignment
# Alias used by env-style configs / ablation CLI (plantvillage | plantdoc | demo | sampletest_* | realistic)
DATASET = "plantvillage"

# Number of output classes per dataset
NUM_CLASSES_PLANTVILL = 38          # PlantVillage: 38 disease + healthy classes
NUM_CLASSES_PLANTDOC = 27           # PlantDoc: 27 classes (real-world, harder benchmark)
NUM_CLASSES_DEMO = 5                # Tiny synthetic demo set (no download needed)
NUM_CLASSES_SAMPLETEST_PV = 38      # SampleTest_PlantVillage (synthetic, official 38 names)
NUM_CLASSES_SAMPLETEST_PD = 27      # SampleTest_PlantDoc (synthetic, official 27 names)
NUM_CLASSES_REALISTIC_PV = 38       # RealisticPV (harder synthetic, official 38 names)
NUM_CLASSES_BALANCED_PV = 38        # BalancedPV (medium synthetic, official 38 names)

# Canonical on-disk folder names recognised by resolve_dataset_name()
CANONICAL_DATASETS = (
    "PlantVillage",
    "PlantDoc",
    "Demo",
    "SampleTest_PlantVillage",
    "SampleTest_PlantDoc",
    "RealisticPV",
    "BalancedPV",
)

# Strings accepted by argparse --dataset / --dataset_name (canonical + aliases)
DATASET_CLI_CHOICES = (
    "PlantVillage",
    "PlantDoc",
    "Demo",
    "SampleTest_PlantVillage",
    "SampleTest_PlantDoc",
    "plantvillage",
    "plantdoc",
    "demo",
    "sampletest",
    "sample_test",
    "sampletest_plantvillage",
    "sampletest_plantdoc",
    "RealisticPV",
    "realistic",
    "realisticpv",
    "realistic_pv",
    "BalancedPV",
    "balanced",
    "balancedpv",
    "balanced_pv",
)


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

# Ablation variant: baseline | multiscale | attention | multiscale_attention
MODEL_VARIANT = "baseline"

SUPPORTED_VARIANTS = (
    "baseline",
    "multiscale",
    "attention",
    "multiscale_attention",
)


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

def resolve_dataset_name(dataset_name: str) -> str:
    """Normalise dataset aliases to the on-disk folder name.

    Bare ``sampletest`` / ``sample_test`` map to ``SampleTest_PlantVillage``
    (38 classes). Prefer ``sampletest_plantvillage`` or
    ``sampletest_plantdoc`` when you need to be explicit.

    ``realistic`` / ``realisticpv`` / ``realistic_pv`` map to ``RealisticPV``
    (38 classes, harder synthetic set).

    ``balanced`` / ``balancedpv`` / ``balanced_pv`` map to ``BalancedPV``
    (38 classes, medium synthetic set for demo ablations).
    """
    mapping = {
        "plantvillage": "PlantVillage",
        "plantdoc": "PlantDoc",
        "demo": "Demo",
        "plant_village": "PlantVillage",
        "plant_doc": "PlantDoc",
        "sampletest": "SampleTest_PlantVillage",
        "sample_test": "SampleTest_PlantVillage",
        "sampletest_plantvillage": "SampleTest_PlantVillage",
        "sample_test_plantvillage": "SampleTest_PlantVillage",
        "sampletestplantvillage": "SampleTest_PlantVillage",
        "sampletest_plantdoc": "SampleTest_PlantDoc",
        "sample_test_plantdoc": "SampleTest_PlantDoc",
        "sampletestplantdoc": "SampleTest_PlantDoc",
        "realistic": "RealisticPV",
        "realisticpv": "RealisticPV",
        "realistic_pv": "RealisticPV",
        "balanced": "BalancedPV",
        "balancedpv": "BalancedPV",
        "balanced_pv": "BalancedPV",
    }
    key = dataset_name.strip()
    if key in CANONICAL_DATASETS:
        return key
    normalised = key.lower().replace("-", "_")
    resolved = mapping.get(normalised)
    if resolved is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {', '.join(CANONICAL_DATASETS)}. "
            f"SampleTest aliases: sampletest / sampletest_plantvillage "
            f"(38 cls), sampletest_plantdoc (27 cls). "
            f"RealisticPV aliases: realistic / realisticpv / realistic_pv (38 cls). "
            f"BalancedPV aliases: balanced / balancedpv / balanced_pv (38 cls)."
        )
    return resolved


def resolve_variant(variant: str) -> str:
    """Normalise a model-variant string to one of ``SUPPORTED_VARIANTS``."""
    key = variant.strip().lower().replace("-", "_")
    aliases = {
        "baseline": "baseline",
        "a": "baseline",
        "multiscale": "multiscale",
        "multi_scale": "multiscale",
        "msff": "multiscale",
        "b": "multiscale",
        "attention": "attention",
        "cbam": "attention",
        "c": "attention",
        "multiscale_attention": "multiscale_attention",
        "multi_scale_attention": "multiscale_attention",
        "proposed": "multiscale_attention",
        "d": "multiscale_attention",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown model variant '{variant}'. "
            f"Supported: {list(SUPPORTED_VARIANTS)}"
        )
    return aliases[key]


def get_num_classes(dataset_name: str) -> int:
    """Return the number of output classes for the specified dataset.

    Args:
        dataset_name: Canonical name or alias (see ``resolve_dataset_name``).

    Returns:
        Integer number of classes.

    Raises:
        ValueError: If *dataset_name* is not recognised.
    """
    canonical = resolve_dataset_name(dataset_name)
    mapping = {
        "PlantVillage": NUM_CLASSES_PLANTVILL,
        "PlantDoc":     NUM_CLASSES_PLANTDOC,
        "Demo":         NUM_CLASSES_DEMO,
        "SampleTest_PlantVillage": NUM_CLASSES_SAMPLETEST_PV,
        "SampleTest_PlantDoc":     NUM_CLASSES_SAMPLETEST_PD,
        "RealisticPV":             NUM_CLASSES_REALISTIC_PV,
        "BalancedPV":              NUM_CLASSES_BALANCED_PV,
    }
    return mapping[canonical]
