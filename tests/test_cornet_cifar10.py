"""
evaluate_cornet_cifar10.py
==========================
Loads a saved CORnet-S checkpoint and evaluates it on the CIFAR-10 test set.

Reports:
  - Per-image true vs predicted label  (with --verbose)
  - Overall accuracy
  - Per-class accuracy
  - Confusion matrix

Usage:
    python evaluate_cornet_cifar10.py                  # summary only
    python evaluate_cornet_cifar10.py --verbose        # + per-image printout
    python evaluate_cornet_cifar10.py --verbose --mistakes-only  # only wrong ones
"""
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from cornet_2026.models import cornet

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — must match exactly what was used during training
# Rule: if your eval transforms differ from training val transforms, your
# accuracy numbers will be wrong. Always keep them in sync.
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE   = 96
BATCH_SIZE = 128
NUM_CLASSES = 10
DATA_DIR   = "./data"
CKPT_PATH  = "./cornet_cifar10_best.pth"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# CLI FLAGS
# Rule: use argparse to make scripts configurable from the command line without
# editing the source code. This is the standard Python way; avoid hardcoding
# behaviour that users might reasonably want to toggle.
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CORnet-S on CIFAR-10")
    parser.add_argument("--verbose", action="store_true",
                        help="Print true vs predicted label for every image")
    parser.add_argument("--mistakes-only", action="store_true",
                        help="When --verbose is set, only print misclassified images")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load the test set only
# Note: NO random augmentations here (no flips, no crops).
# Rule: augmentation is only for training. At eval time you want deterministic,
# unmodified images so results are reproducible and comparable.
# ─────────────────────────────────────────────────────────────────────────────
def get_test_loader(batch_size=BATCH_SIZE):
    test_tf = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_ds = torchvision.datasets.CIFAR10(
        DATA_DIR, train=False, download=False, transform=test_tf
    )
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Rebuild the model architecture, then load saved weights
# Rule: torch.save() only saves the weights (state_dict), NOT the model class.
# You must always rebuild the architecture first, THEN load the weights into it.
# Think of state_dict as the "values" and the model class as the "structure".
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    # Rebuild architecture (must match training exactly)
    wrapper = cornet.cornet_s(pretrained=False)
    if isinstance(wrapper, nn.DataParallel):
        model = wrapper.module
    else:
        model = wrapper[0]

    model.decoder.linear = nn.Linear(
        model.decoder.linear.in_features, NUM_CLASSES
    )

    # Load the saved checkpoint
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    # Rule: always use map_location when loading checkpoints. Without it,
    # a model saved on GPU will crash when loaded on a CPU-only machine.

    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val acc during training: {checkpoint['val_acc']*100:.2f}%)")

    model.to(DEVICE)
    model.eval()  # Rule: always call model.eval() before inference — this
                  # disables dropout and switches BatchNorm to use running stats
                  # instead of batch stats. Forgetting this gives wrong results.
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run evaluation
# We track two things:
#   - overall correct/total  → overall accuracy
#   - per-class correct/total → per-class accuracy (tells you WHERE it fails)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, verbose=False, mistakes_only=False):
    overall_correct = 0
    overall_total   = 0

    # One counter per class
    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES

    # Confusion matrix: rows = true labels, cols = predicted labels
    # confusion[i][j] = number of class-i images predicted as class-j
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    image_idx = 0  # global image counter for per-image printout

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        preds = outputs.argmax(dim=1)  # pick the class with highest score

        # Overall stats
        overall_correct += preds.eq(labels).sum().item()
        overall_total   += labels.size(0)

        # Per-class stats + optional per-image print
        for label, pred in zip(labels, preds):
            true_name = CLASS_NAMES[label.item()]
            pred_name = CLASS_NAMES[pred.item()]
            correct   = (label == pred).item()

            class_correct[label.item()] += correct
            class_total[label.item()]   += 1
            confusion[label.item(), pred.item()] += 1

            # ── per-image printout (mirrors run.py style) ─────────────────
            # Rule: only print when asked — verbose output on 10,000 images
            # floods the terminal and makes logs hard to read by default.
            if verbose:
                if not mistakes_only or not correct:
                    status = "✅ MATCH" if correct else "❌ WRONG"
                    print(f"[{image_idx:05d}] True: {true_name:<12} | "
                          f"Predicted: {pred_name:<12} | {status}")

            image_idx += 1

    return overall_correct, overall_total, class_correct, class_total, confusion


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Print results clearly
# ─────────────────────────────────────────────────────────────────────────────
def print_results(overall_correct, overall_total, class_correct, class_total, confusion):
    overall_acc = overall_correct / overall_total * 100
    print(f"\n{'='*50}")
    print(f"  Overall Test Accuracy: {overall_acc:.2f}%  ({overall_correct}/{overall_total})")
    print(f"{'='*50}")

    print(f"\n{'Per-Class Accuracy':}")
    print(f"  {'Class':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*42}")
    for i, name in enumerate(CLASS_NAMES):
        acc = class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
        print(f"  {name:<12} {class_correct[i]:>8} {class_total[i]:>8} {acc:>9.2f}%")

    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"  {'':12}", end="")
    for name in CLASS_NAMES:
        print(f"{name[:6]:>8}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<12}", end="")
        for j in range(NUM_CLASSES):
            print(f"{confusion[i,j].item():>8}", end="")
        print()

    # Surface the most confused pairs — useful for understanding failure modes
    print(f"\nTop 5 most confused pairs (true → predicted):")
    off_diag = confusion.clone()
    for i in range(NUM_CLASSES):
        off_diag[i, i] = 0  # zero out correct predictions
    flat = off_diag.view(-1)
    top5 = flat.topk(5).indices
    for idx in top5:
        true_cls = idx.item() // NUM_CLASSES
        pred_cls = idx.item() %  NUM_CLASSES
        count    = off_diag[true_cls, pred_cls].item()
        print(f"  {CLASS_NAMES[true_cls]:<12} → {CLASS_NAMES[pred_cls]:<12}  ({count} times)")


def main():
    args = parse_args()

    print(f"Device: {DEVICE}")

    # Rule: in verbose mode we use batch_size=1 so we can print one image at a
    # time with its index. With batch_size=128 we'd only know the batch, not
    # the individual image position within it.
    batch_size  = 1 if args.verbose else BATCH_SIZE
    test_loader = get_test_loader(batch_size=batch_size)
    model       = load_model(CKPT_PATH)

    if args.verbose:
        mode = "mistakes only" if args.mistakes_only else "all images"
        print(f"\nVerbose mode ON ({mode})\n")

    print("Running evaluation on test set...")
    results = evaluate(model, test_loader,
                       verbose=args.verbose,
                       mistakes_only=args.mistakes_only)
    print_results(*results)


if __name__ == "__main__":
    main()