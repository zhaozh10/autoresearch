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

TIME_BUDGET = 2400        # training time budget in seconds (40 minutes)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join("/mnt/ocean_storage/data")
DATA_DIR = os.path.join(BASE_DIR, "VinDr-Mammo")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
