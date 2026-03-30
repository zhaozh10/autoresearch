import os

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify after creation)
# ---------------------------------------------------------------------------

TIME_BUDGET = 1800        # training time budget in seconds (30 minutes)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = "/mnt/ocean_storage/users/zzhao"
DATA_DIR = os.path.join(BASE_DIR, "ISIC2019")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "ISIC_2019_Training_Input")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "ISIC_2019_Training_GroundTruth.csv")
TRAIN_META_CSV = os.path.join(DATA_DIR, "ISIC_2019_Training_Metadata.csv")

TEST_IMG_DIR = os.path.join(DATA_DIR, "ISIC_2019_Test_Input")
TEST_LABELS_CSV = os.path.join(DATA_DIR, "ISIC_2019_Test_GroundTruth.csv")
TEST_META_CSV = os.path.join(DATA_DIR, "ISIC_2019_Test_Metadata.csv")

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = len(CLASS_NAMES)  # 8

PRIMARY_METRIC = "balanced_accuracy"
METRIC_DIRECTION = "higher"  # higher is better

SEED = 42
VAL_FRACTION = 0.20  # 20% of training data for validation
