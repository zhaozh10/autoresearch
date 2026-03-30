"""
ISIC 2019 skin lesion classification — training script.
Usage: uv run train.py
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

import timm

import prepare
from data import get_dataloaders

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE = 48
LR = 3e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 3

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model():
    model = timm.create_model(
        "efficientnet_b2",
        pretrained=True,
        num_classes=prepare.NUM_CLASSES,
        drop_rate=0.3,
        drop_path_rate=0.2,
    )
    return model

# ---------------------------------------------------------------------------
# Evaluation  (FROZEN — do not change unless user asks)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Compute all metrics on a dataloader. Returns dict with primary_metric and metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Balanced accuracy (primary metric)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    # Weighted AUC (one-vs-rest)
    try:
        one_hot = np.eye(prepare.NUM_CLASSES)[all_labels]
        auc_weighted = roc_auc_score(one_hot, all_probs, average="weighted", multi_class="ovr")
    except ValueError:
        auc_weighted = 0.0

    # F1 macro
    f1_mac = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class sensitivity (recall)
    sensitivities = []
    for c in range(prepare.NUM_CLASSES):
        mask = all_labels == c
        if mask.sum() > 0:
            sensitivities.append((all_preds[mask] == c).mean())
        else:
            sensitivities.append(0.0)
    sensitivity_macro = float(np.mean(sensitivities))

    return {
        "primary_metric": bal_acc,
        "metrics": {
            "balanced_accuracy": bal_acc,
            "auc_weighted": auc_weighted,
            "f1_macro": f1_mac,
            "sensitivity_macro": sensitivity_macro,
        },
    }

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, class_weights = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    class_weights = class_weights.to(device)

    # Model
    model = build_model().to(device)
    print(f"Model: efficientnet_b2 | Params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Warmup + cosine schedule
    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        # cosine decay after warmup
        progress = (step - warmup_steps) / max(1, 50 * steps_per_epoch - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpointing
    os.makedirs(prepare.CHECKPOINT_DIR, exist_ok=True)
    best_metric = -float("inf")
    best_path = os.path.join(prepare.CHECKPOINT_DIR, "best.pt")
    last_path = os.path.join(prepare.CHECKPOINT_DIR, "last.pth")

    # --------------- Training loop ---------------
    total_start = time.time()
    train_start = None
    epoch = 0
    global_step = 0

    while True:
        epoch += 1
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            if train_start is None:
                train_start = time.time()

            # Check time budget
            elapsed = time.time() - train_start
            if elapsed >= prepare.TIME_BUDGET:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1

        # Check time budget after epoch
        if train_start is not None:
            elapsed = time.time() - train_start
            if elapsed >= prepare.TIME_BUDGET:
                print(f"Time budget reached after epoch {epoch} ({elapsed:.1f}s)")
                break

        avg_loss = running_loss / max(n_batches, 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} | loss={avg_loss:.4f} | lr={cur_lr:.2e} | elapsed={elapsed:.0f}s")

        # Save last checkpoint
        torch.save(model.state_dict(), last_path)

        # Evaluate
        results = evaluate(model, val_loader, device)
        pm = results["primary_metric"]
        metrics = results["metrics"]
        print(f"  val balanced_accuracy={pm:.4f} | auc={metrics['auc_weighted']:.4f} | f1={metrics['f1_macro']:.4f}")

        # Save best
        if pm > best_metric:
            best_metric = pm
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best: {pm:.4f}")

    # --------------- Final report ---------------
    training_seconds = time.time() - (train_start or total_start)
    total_seconds = time.time() - total_start

    # Peak VRAM
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_vram_mb = 0.0

    # Load best and evaluate
    model.load_state_dict(torch.load(best_path, weights_only=True))
    results = evaluate(model, val_loader, device)
    metrics = results["metrics"]

    print("\n---")
    print(f"balanced_accuracy:      {metrics['balanced_accuracy']:.4f}")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}:            {v:.4f}")
    print(f"\ntraining_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")


if __name__ == "__main__":
    main()
