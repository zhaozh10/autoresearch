"""
ISIC 2019: Melanoma vs Nevus binary classification.
DDP training with timm pretrained backbone.
Usage: uv run train.py  (auto-launches DDP if multiple GPUs available)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import timm
from sklearn.metrics import roc_auc_score, roc_curve

from prepare import TIME_BUDGET, SEED, IMG_SIZE
from data import get_dataloaders

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BACKBONE = "efficientnet_b3"
PRETRAINED = True
BATCH_SIZE = 64          # per GPU
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_FRAC = 0.05       # fraction of time budget for LR warmup
NUM_WORKERS = 4

# ---------------------------------------------------------------------------
# Evaluation function (DO NOT MODIFY unless user requests)
# ---------------------------------------------------------------------------

def evaluate(model, val_loader, device):
    """Evaluate model. Primary metric: mel_sensitivity at Youden-optimal threshold."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).float().numpy()
    labels = torch.cat(all_labels).float().numpy()

    # AUC
    auc = roc_auc_score(labels, probs)

    # Youden-optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = float(thresholds[optimal_idx])

    # Metrics at optimal threshold
    preds = (probs >= optimal_threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "primary_metric": float(sensitivity),
        "metrics": {
            "mel_sensitivity": float(sensitivity),
            "mel_specificity": float(specificity),
            "auc": float(auc),
            "f1": float(f1),
            "threshold": float(optimal_threshold),
        },
    }

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(progress, base_lr):
    """Cosine decay with linear warmup, based on time progress [0, 1]."""
    if progress < WARMUP_FRAC:
        return base_lr * progress / WARMUP_FRAC
    else:
        decay = (progress - WARMUP_FRAC) / (1.0 - WARMUP_FRAC)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * decay))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")

    # DDP setup
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda:0")

    is_main = local_rank == 0

    # Experiment directory
    exp_dir = os.path.join("runs", time.strftime("%Y%m%d_%H"))
    if is_main:
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    # Data
    train_loader, val_loader, train_sampler, pos_weight = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, distributed=distributed,
    )
    pos_weight = pos_weight.to(device)

    if is_main:
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
        print(f"Pos weight (NV/MEL ratio): {pos_weight.item():.2f}")
        print(f"Backbone: {BACKBONE}, IMG_SIZE: {IMG_SIZE}, BATCH_SIZE: {BATCH_SIZE}")
        print(f"LR: {LR}, WEIGHT_DECAY: {WEIGHT_DECAY}")
        print(f"Time budget: {TIME_BUDGET}s")

    # Model
    model = timm.create_model(BACKBONE, pretrained=PRETRAINED, num_classes=1)
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop
    t_start = time.time()
    best_primary = -1.0
    best_metrics = {}
    epoch = 0

    while True:
        elapsed = time.time() - t_start
        if elapsed >= TIME_BUDGET and epoch > 0:
            break

        if distributed:
            train_sampler.set_epoch(epoch)

        # Update LR based on time progress
        progress = min(elapsed / TIME_BUDGET, 1.0)
        lr_now = get_lr(progress, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Train one epoch
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            if time.time() - t_start >= TIME_BUDGET and epoch > 0:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images).squeeze(-1)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)

        # Evaluate on rank 0
        if is_main:
            raw_model = model.module if distributed else model
            results = evaluate(raw_model, val_loader, device)
            primary = results["primary_metric"]
            metrics = results["metrics"]

            elapsed_now = time.time() - t_start
            print(
                f"Epoch {epoch:3d} | loss={avg_loss:.4f} | lr={lr_now:.2e} | "
                f"sens={metrics['mel_sensitivity']:.4f} | "
                f"spec={metrics['mel_specificity']:.4f} | "
                f"auc={metrics['auc']:.4f} | "
                f"f1={metrics['f1']:.4f} | "
                f"thr={metrics['threshold']:.3f} | "
                f"elapsed={elapsed_now:.0f}s"
            )

            # Checkpoint: last
            ckpt = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "primary_metric": primary,
                "metrics": metrics,
            }
            torch.save(ckpt, os.path.join(exp_dir, "checkpoints", "last.pth"))

            # Checkpoint: best (by mel_sensitivity)
            if primary > best_primary:
                best_primary = primary
                best_metrics = metrics.copy()
                torch.save(ckpt, os.path.join(exp_dir, "checkpoints", "best.pth"))
                print(f"  -> New best mel_sensitivity: {primary:.4f}")

        epoch += 1

    # Synchronize before summary
    if distributed:
        dist.barrier()

    # Final summary (rank 0 only)
    if is_main:
        t_end = time.time()
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        print()
        print("---")
        print(f"mel_sensitivity:  {best_primary:.6f}")
        print()
        for k, v in best_metrics.items():
            print(f"{k}:  {v:.6f}")
        print()
        print(f"training_seconds: {t_end - t_start:.1f}")
        print(f"total_seconds:    {t_end - t_start:.1f}")
        print(f"peak_vram_mb:     {peak_vram_mb:.1f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            import subprocess
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={num_gpus}",
                os.path.abspath(sys.argv[0]),
            ]
            sys.exit(subprocess.run(cmd).returncode)
    main()
