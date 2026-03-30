import os
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 1800        # training time budget in seconds (30 minutes)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join("/mnt/ocean_storage/users/zzhao")
DATA_DIR = os.path.join(BASE_DIR, "ISIC2019")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
