"""
Data loading and preprocessing for ISIC 2019 MEL vs NV classification.
Lesion-level splitting to prevent data leakage.
"""

import json
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as T

from prepare import DATA_DIR, IMG_DIR, TRAIN_GT, TRAIN_META, IMG_SIZE, SEED, VAL_RATIO

SPLIT_FILE = os.path.join(os.path.dirname(__file__), "split.json")


class ISICDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, f"{row['image']}.jpg")).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label


def get_transforms(is_train):
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(int(IMG_SIZE * 1.14)),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def _build_dataframe():
    """Load ISIC 2019, filter to MEL + NV, merge lesion IDs."""
    gt = pd.read_csv(TRAIN_GT)
    meta = pd.read_csv(TRAIN_META)

    mask = (gt["MEL"] == 1.0) | (gt["NV"] == 1.0)
    gt = gt[mask].copy()
    gt["label"] = gt["MEL"].astype(int)  # MEL=1 (positive), NV=0

    df = gt.merge(meta[["image", "lesion_id"]], on="image", how="left")

    missing = df["lesion_id"].isna()
    df.loc[missing, "lesion_id"] = [f"_miss_{i}" for i in range(missing.sum())]
    return df


def load_splits():
    """Load split from split.json if it exists, otherwise create and save it."""
    df = _build_dataframe()

    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f:
            saved = json.load(f)
        train_images = set(saved["train"])
        val_images = set(saved["val"])
        train_df = df[df["image"].isin(train_images)]
        val_df = df[df["image"].isin(val_images)]
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
        train_idx, val_idx = next(splitter.split(df, df["label"], groups=df["lesion_id"]))
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        # Persist to disk
        split_data = {
            "train": train_df["image"].tolist(),
            "val": val_df["image"].tolist(),
        }
        with open(SPLIT_FILE, "w") as f:
            json.dump(split_data, f)

    return train_df, val_df


def get_dataloaders(batch_size, num_workers=8, distributed=False):
    """Create train and val dataloaders.

    Train loader uses DistributedSampler if distributed=True.
    Val loader is always non-distributed (evaluated on rank 0 only).
    """
    train_df, val_df = load_splits()

    train_ds = ISICDataset(train_df, IMG_DIR, get_transforms(is_train=True))
    val_ds = ISICDataset(val_df, IMG_DIR, get_transforms(is_train=False))

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Class weights for BCEWithLogitsLoss
    n_mel = int((train_df["label"] == 1).sum())
    n_nv = int((train_df["label"] == 0).sum())
    pos_weight = torch.tensor([n_nv / n_mel], dtype=torch.float32)

    return train_loader, val_loader, train_sampler, pos_weight
