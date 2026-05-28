"""
train_cornet_cifar10.py
=======================
Retrains CORnet-S on CIFAR-10.
Usage:
    pip install torch torchvision cornet
    python train_cornet_cifar10.py
"""

import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import CORnet from the local package layout
from src.cornet_2026.models import cornet

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (single source of truth — change config here, not scattered code)
# ─────────────────────────────────────────────────────────────────────────────
# Why 96? CORnet-S uses strides of 2 in V1 (twice), then stride-2 again in
# each subsequent area. 32×32 collapses to a 1×1 map too early.
# 96 → 48 → 24 → 12 → 6 → valid feature maps everywhere.
IMG_SIZE    = 96
NUM_CLASSES = 10
BATCH_SIZE  = 128
EPOCHS      = 50
LR          = 0.01
MOMENTUM    = 0.9
WEIGHT_DECAY = 1e-4
DATA_DIR    = "./data"
CKPT_PATH   = "./cornet_cifar10_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA
# Rule: always apply normalization AFTER ToTensor(), because ToTensor()
# converts PIL images (0-255) to float tensors in [0, 1], which is what
# Normalize() expects. Swapping the order produces nonsensical values.
# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 channel means/stds (pre-computed over the training set)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def get_dataloaders():
    train_tf = T.Compose([
        T.Resize(IMG_SIZE),            # upsample 32→96 to satisfy CORnet strides
        T.RandomHorizontalFlip(),      # cheap data augmentation
        T.RandomCrop(IMG_SIZE, padding=8),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    val_tf = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=False, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=val_tf)
    
    # Rule: num_workers > 0 moves data loading off the main thread so the GPU
    # is never waiting idle for the CPU. pin_memory=True speeds up CPU→GPU transfer.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MODEL
# CORnet-S returns a nn.Sequential of (model, time_steps).
# We unwrap it and swap the final linear layer (1000 ImageNet classes → 10).
# ─────────────────────────────────────────────────────────────────────────────
def build_model():
    # cornet.cornet_s() returns a nn.Sequential wrapping the actual model
    # The actual CORnet model is at index 0
    wrapper = cornet.cornet_s(pretrained=False)

# Handle both DataParallel and plain Sequential packaging
    if isinstance(wrapper, nn.DataParallel):
        model = wrapper.module  # unwrap DataParallel → inner model
    else:
        model = wrapper[0]      # unwrap Sequential → inner model

    # Inspect the decoder to find its linear layer
    # CORnet-S structure: V1 → V2 → V4 → IT → decoder
    # decoder is an nn.Sequential with a Linear at the end
    decoder      = model.decoder
    in_features  = decoder.linear.in_features   # 512 in CORnet-S

    # Rule: when adapting a pretrained/pre-built model to a new number of
    # classes, replace ONLY the final classification head. All earlier layers
    # keep their learned feature representations.
    decoder.linear = nn.Linear(in_features, NUM_CLASSES)

    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAINING & VALIDATION LOOPS
# Separating train/validate into functions keeps each function doing ONE thing.
# This is the Single Responsibility Principle — makes debugging easier.
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()          # Rule: always zero gradients BEFORE forward
        outputs = model(images)        # CORnet returns (output, hidden) tuple
        # Rule: CORnet-S returns a named tuple; the classification output is [0]
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # ── bookkeeping ──────────────────────────────────────────────────────
        total_loss += loss.item() * images.size(0)  # accumulate un-averaged loss
        preds       = outputs.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += images.size(0)

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()   # Rule: disable gradient tracking during validation — saves
                   # memory and speeds up inference (no computation graph built)
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss        = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    train_loader, val_loader = get_dataloaders()
    model     = build_model()
    criterion = nn.CrossEntropyLoss()

    # SGD + momentum is standard for training CNNs from scratch.
    # Rule: Adam converges faster but often generalises worse than SGD+momentum
    # for image classification. Use Adam for fine-tuning, SGD for scratch training.
    optimizer = SGD(model.parameters(), lr=LR,
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # CosineAnnealing smoothly decays LR from LR → 0 over T_max epochs.
    # This avoids sharp LR drops that can destabilise training.
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc     = validate(model, val_loader, criterion)

        scheduler.step()   # Rule: call scheduler.step() AFTER optimizer.step()
                           # (since PyTorch 1.1, calling it before raises a warning)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{EPOCHS} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%")

        # Save the best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
                "optimizer":  optimizer.state_dict(),
            }, CKPT_PATH)
            print(f"  ✔ New best model saved ({val_acc*100:.2f}%)")

    print(f"\nTraining complete. Best val acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved to: {CKPT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Rule: always guard script entry with `if __name__ == "__main__":`
# This prevents the training loop from running when this file is imported
# as a module elsewhere (e.g., for unit tests or inference scripts).
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()