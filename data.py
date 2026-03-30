import os
import csv
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import prepare

# ---------------------------------------------------------------------------
# Read labels CSV → list of (image_id, class_index)
# ---------------------------------------------------------------------------

def load_labels(csv_path):
    """Return list of (image_id, class_idx) from a one-hot ground truth CSV."""
    samples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image"]
            for idx, cls in enumerate(prepare.CLASS_NAMES):
                if float(row[cls]) == 1.0:
                    samples.append((image_id, idx))
                    break
    return samples

# ---------------------------------------------------------------------------
# Read metadata CSV → dict image_id → lesion_id (may be empty string)
# ---------------------------------------------------------------------------

def load_lesion_ids(csv_path):
    mapping = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["image"]] = row.get("lesion_id", "").strip()
    return mapping

# ---------------------------------------------------------------------------
# Lesion-level stratified split
# ---------------------------------------------------------------------------

def split_train_val(samples, lesion_map, val_frac=0.20, seed=42):
    """Split samples into train/val at the lesion level (patient-level proxy)."""
    rng = random.Random(seed)

    groups_by_class = defaultdict(list)
    _counter = 0
    lesion_to_key = {}
    for image_id, cls_idx in samples:
        lid = lesion_map.get(image_id, "")
        if lid:
            key = lid
        else:
            key = f"__solo_{_counter}"
            _counter += 1
        if key not in lesion_to_key:
            lesion_to_key[key] = (key, cls_idx, [])
        lesion_to_key[key][2].append((image_id, cls_idx))

    for key, cls_idx, samps in lesion_to_key.values():
        groups_by_class[cls_idx].append((key, samps))

    train_samples, val_samples = [], []

    for cls_idx in range(prepare.NUM_CLASSES):
        groups = groups_by_class[cls_idx]
        rng.shuffle(groups)
        total_images = sum(len(s) for _, s in groups)
        target_val = int(total_images * val_frac)
        val_count = 0
        for key, samps in groups:
            if val_count < target_val:
                val_samples.extend(samps)
                val_count += len(samps)
            else:
                train_samples.extend(samps)

    return train_samples, val_samples

# ---------------------------------------------------------------------------
# In-memory image cache  (load once from network FS, then train from RAM)
# ---------------------------------------------------------------------------

def _load_one(args):
    image_id, img_dir, size = args
    img_path = os.path.join(img_dir, f"{image_id}.jpg")
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return image_id, img

def preload_images(image_ids, img_dir, size, max_workers=16):
    """Load and resize all images into a dict {image_id: PIL.Image}."""
    cache = {}
    tasks = [(iid, img_dir, size) for iid in image_ids]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, (iid, img) in enumerate(pool.map(_load_one, tasks)):
            cache[iid] = img
            if (i + 1) % 5000 == 0:
                print(f"  cached {i+1}/{len(tasks)} images ({time.time()-t0:.0f}s)")
    print(f"  cached {len(cache)} images in {time.time()-t0:.0f}s")
    return cache

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMG_SIZE = 384  # Higher resolution for dermoscopy detail

class ISICDataset(Dataset):
    def __init__(self, samples, transform=None, cache=None):
        self.samples = samples
        self.transform = transform
        self.cache = cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, label = self.samples[idx]
        img = self.cache[image_id].copy()  # copy so transforms don't mutate cache
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------------------------------------------------------------------------
# Transforms  (no Resize — images are pre-resized in cache)
# ---------------------------------------------------------------------------

def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ---------------------------------------------------------------------------
# Compute class weights for weighted loss
# ---------------------------------------------------------------------------

def compute_class_weights(samples):
    counts = np.zeros(prepare.NUM_CLASSES, dtype=np.float64)
    for _, cls_idx in samples:
        counts[cls_idx] += 1
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * prepare.NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)

# ---------------------------------------------------------------------------
# Build dataloaders
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size=32, num_workers=4):
    """Return train_loader, val_loader, class_weights tensor."""
    all_samples = load_labels(prepare.TRAIN_LABELS_CSV)
    lesion_map = load_lesion_ids(prepare.TRAIN_META_CSV)

    train_samples, val_samples = split_train_val(
        all_samples, lesion_map,
        val_frac=prepare.VAL_FRACTION,
        seed=prepare.SEED,
    )

    # Pre-load all images into RAM (eliminates network I/O during training)
    all_ids = list(set(iid for iid, _ in train_samples + val_samples))
    print(f"Pre-loading {len(all_ids)} images at {IMG_SIZE}x{IMG_SIZE}...")
    cache = preload_images(all_ids, prepare.TRAIN_IMG_DIR, IMG_SIZE)

    train_ds = ISICDataset(train_samples, get_train_transform(), cache)
    val_ds = ISICDataset(val_samples, get_val_transform(), cache)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True,
    )

    class_weights = compute_class_weights(train_samples)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, class_weights
