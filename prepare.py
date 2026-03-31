import os

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 2400  # training time budget in seconds (40 minutes)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

BASE_DIR = "/mnt/ocean_storage/users/zzhao"
DATA_DIR = os.path.join(BASE_DIR, "ISIC2019")
IMG_DIR = os.path.join(DATA_DIR, "ISIC_2019_Training_Input")
TRAIN_GT = os.path.join(DATA_DIR, "ISIC_2019_Training_GroundTruth.csv")
TRAIN_META = os.path.join(DATA_DIR, "ISIC_2019_Training_Metadata.csv")

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

CLASSES = ["NV", "MEL"]  # index 0 = NV (negative), index 1 = MEL (positive)
NUM_CLASSES = 2
IMG_SIZE = 384
SEED = 42
VAL_RATIO = 0.2
