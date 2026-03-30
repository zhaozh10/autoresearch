import os
import csv
import random
from collections import defaultdict

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
    """Split samples into train/val at the lesion level (patient-level proxy).

    Images sharing a lesion_id always go to the same split.
    Images without a lesion_id are each treated as their own unique group.
    Stratified by class to preserve balance.
    """
    rng = random.Random(seed)

    # Group samples by lesion_id (or unique placeholder)
    # Each group: (lesion_key, class_idx, [list of samples])
    groups_by_class = defaultdict(list)  # class_idx -> list of (key, [samples])
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

    # Organise groups by their majority class
    for key, cls_idx, samps in lesion_to_key.values():
        groups_by_class[cls_idx].append((key, samps))

    train_samples, val_samples = [], []

    for cls_idx in range(prepare.NUM_CLASSES):
        groups = groups_by_class[cls_idx]
        rng.shuffle(groups)
        # Count total images in this class
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
# Dataset
# ---------------------------------------------------------------------------

class ISICDataset(Dataset):
    def __init__(self, samples, img_dir, transform=None):
        """
        samples: list of (image_id, class_idx)
        img_dir: path to image directory
        transform: torchvision transform
        """
        self.samples = samples
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMG_SIZE = 512

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
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
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ---------------------------------------------------------------------------
# Compute class weights for weighted loss
# ---------------------------------------------------------------------------

def compute_class_weights(samples):
    """Return a tensor of inverse-frequency weights for each class."""
    counts = np.zeros(prepare.NUM_CLASSES, dtype=np.float64)
    for _, cls_idx in samples:
        counts[cls_idx] += 1
    # inverse frequency, normalised so weights sum to NUM_CLASSES
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

    train_ds = ISICDataset(train_samples, prepare.TRAIN_IMG_DIR, get_train_transform())
    val_ds = ISICDataset(val_samples, prepare.TRAIN_IMG_DIR, get_val_transform())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_weights = compute_class_weights(train_samples)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, class_weights
