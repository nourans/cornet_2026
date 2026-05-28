"""
cw_helperfxnsALL.py
Helper functions for the Carlini & Wagner (C&W) L2 attack.

Key conceptual difference from FGSM:
  - FGSM  : one-shot gradient step, controlled by epsilon (L-inf bound)
  - C&W   : iterative optimisation that FINDS the smallest perturbation
             that causes a mis-classification, using a custom loss that
             balances two objectives:
               (1) minimise ||delta||_2  (keep perturbation small)
               (2) make the model confident on a WRONG class

Because C&W is iterative (many forward+backward passes per image),
it is much slower than FGSM but produces near-imperceptible adversarial
examples even at low confidence targets.

Key hyperparameters:
  c          – trade-off constant: higher = more attack pressure, less
               care about perturbation magnitude
  kappa      – confidence margin: how much MORE confident the model must
               be on the wrong class than on the true class
  lr         – learning rate for the Adam optimiser
  num_steps  – number of optimisation steps per image
"""

import os
import json
import uuid

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD  = [0.5, 0.5, 0.5]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE I/O
# ─────────────────────────────────────────────────────────────────────────────

def get_all_image_paths(root_dir):
    """Recursively collect every image file under root_dir."""
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                paths.append(os.path.join(dirpath, fname))
    return paths


def get_input_batch(device, filename, preprocess):
    """
    Load a single image, apply preprocess, add a batch dimension,
    and send to device.

    Rule: PIL images are H×W×C in [0,255].  ToTensor() converts to
    C×H×W in [0.0, 1.0].  Normalize() then shifts to the model's
    expected distribution. You must do this in order.
    """
    img = Image.open(filename).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)  # shape: [1, C, H, W]
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def output_prediction(model, input_batch):
    """Return the top-1 predicted class index (integer)."""
    with torch.no_grad():
        logits = model(input_batch)
    return logits.argmax(dim=1).item()


def compare_labels(predicted_index, true_index):
    """Return True if predicted_index == true_index."""
    return int(predicted_index) == int(true_index)


# ─────────────────────────────────────────────────────────────────────────────
# LABEL EXTRACTION  –  ImageNet
# ─────────────────────────────────────────────────────────────────────────────

def extract_true_label(filename):
    """
    For ImageNet folder layout:
        val/n01749939_green_mamba/ILSVRC2012_val_00040948.JPEG
    Returns (class_index: int, human_label: str).
    """
    category_folder = os.path.basename(os.path.dirname(filename))
    wnid = category_folder.split("_")[0]          # e.g. "n01749939"
    human_label = category_folder.split("_", 1)[-1]  # e.g. "green_mamba"

    with open("imagenet_class_index.json") as f:
        idx_to_label = json.load(f)               # {"0": ["n01440764","tench"], ...}

    for idx, (synset, _) in idx_to_label.items():
        if synset == wnid:
            return int(idx), human_label

    raise ValueError(f"WNID {wnid} not found in imagenet_class_index.json")


# ─────────────────────────────────────────────────────────────────────────────
# LABEL EXTRACTION  –  CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────

def extract_true_label_cifar(filename):
    """
    For CIFAR-10 folder layout:
        cifar10_jpegs/test/cat/00123_cat.jpeg
    Returns (class_index: int, class_name: str).
    """
    folder = os.path.basename(os.path.dirname(filename)).lower()
    if folder not in CIFAR10_CLASSES:
        raise ValueError(f"Folder '{folder}' is not a CIFAR-10 class.")
    return CIFAR10_CLASSES.index(folder), folder


# ─────────────────────────────────────────────────────────────────────────────
# SAVE ADVERSARIAL IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def save_adv_image(img_tensor, c_value, true_label, true_key,
                   pred_before, pred_after,
                   output_dir="adv_outputs_cw",
                   mean=None, std=None):
    """
    Denormalise the adversarial tensor back to [0,1] pixel space and
    save as a JPEG.

    Rule: the model works in normalised space (mean≈0, std≈1), but an
    image file must be in [0,1] (or [0,255]).  You have to invert the
    normalisation before saving, otherwise the saved file will look
    completely wrong (very dark or clipped).

    Inverse normalisation formula:
        x_original = x_normalised * std + mean
    which in transform notation is:
        Normalize(mean = [-m/s for m,s in zip(mean,std)],
                  std  = [1/s  for s in std])
    """
    mean = mean or IMAGENET_MEAN
    std  = std  or IMAGENET_STD

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std]
    )

    os.makedirs(output_dir, exist_ok=True)
    img = inv_normalize(img_tensor.squeeze(0).cpu())
    img = img.clamp(0, 1)
    pil_img = transforms.ToPILImage()(img)

    uid = uuid.uuid4().hex[:8]
    filename = (
        f"c{c_value}_{true_label}_true{true_key}"
        f"_before{pred_before}_after{pred_after}_{uid}.jpeg"
    )
    pil_img.save(os.path.join(output_dir, filename))


# ─────────────────────────────────────────────────────────────────────────────
# CARLINI & WAGNER  L2  ATTACK  (core logic)
# ─────────────────────────────────────────────────────────────────────────────
#
# Why C&W works differently from FGSM:
#
# FGSM asks: "given the gradient of the loss, which direction maximises
# the error in one step?"  Answer: sign(grad) * epsilon.
#
# C&W asks a harder question: "what is the SMALLEST perturbation delta
# such that the model is fooled by AT LEAST kappa confidence?"
#
# To solve this optimisation problem it:
#   1. Re-parameterises the image as w (tanh-space) so that the adversarial
#      example x+delta stays in [0,1] pixel space automatically —
#      no need for a separate clipping step.
#   2. Defines a custom loss:
#         L = ||delta||_2^2  +  c * f(x+delta)
#      where f() is 0 when the attack succeeds with kappa margin, and
#      positive otherwise.
#   3. Minimises L with Adam over num_steps iterations.
#
# The tanh reparametrisation trick:
#   w = arctanh(2*x - 1)    (x in [0,1] → w in (-∞, +∞))
#   x_adv = (tanh(w+delta_w) + 1) / 2   stays in (0,1) for any delta_w
#
# This is why C&W doesn't need an epsilon — the constraint is implicit
# in the L2 term of the loss.

def cw_l2_attack(model, input_batch, true_index, device,
                 c=1e-2, kappa=0.0, lr=1e-2, num_steps=100):
    """
    Carlini & Wagner L2 adversarial attack.

    Args:
        model       : PyTorch model (in eval mode, no grad tracking needed
                      on the model itself — we optimise the INPUT)
        input_batch : normalised image tensor  [1, C, H, W]  on device
        true_index  : integer class index of the ground-truth label
        device      : torch.device
        c           : trade-off constant (higher → more attack pressure)
        kappa       : confidence margin (0 = just flip the prediction,
                      higher = flip with more confidence)
        lr          : Adam learning rate
        num_steps   : optimisation iterations

    Returns:
        perturbed_tensor : adversarial image in the SAME normalised space
                           as input_batch  (ready to feed back to the model)
        pred_after       : predicted class index on the adversarial image
    """
    # ── Step 1: convert normalised image → [0,1] pixel space ─────────────
    # We need to work in pixel space for the tanh trick, but inputs arrive
    # normalised.  Invert the normalisation first.
    # (We undo this at the very end to return a normalised tensor.)
    #
    # NOTE: we don't have mean/std here.  The caller is responsible for
    # passing an input_batch that was preprocessed consistently.
    # We operate purely on the tensor values.

    # Detach so no gradients flow back into the original graph
    x0 = input_batch.detach().clone()          # [1, C, H, W], still normalised

    # ── Step 2: initialise the optimisation variable w ────────────────────
    # We optimise w directly; the adversarial example is recovered from w.
    # Initialise w = x0  (start from the clean image, zero perturbation).
    w = x0.clone().requires_grad_(True)
    # Rule: requires_grad_(True) tells PyTorch to track operations on w
    # and compute gradients w.r.t. it during loss.backward().

    optimiser = torch.optim.Adam([w], lr=lr)

    best_adv   = x0.clone()
    best_l2    = float("inf")
    best_pred  = output_prediction(model, x0)

    for step in range(num_steps):
        optimiser.zero_grad()

        # ── Compute L2 distance between w and original ────────────────
        # This is ||delta||_2^2 = ||w - x0||_2^2
        l2_loss = torch.sum((w - x0) ** 2)

        # ── Compute the attack loss f(x_adv) ─────────────────────────
        # We use the f6 formulation from the C&W paper:
        #   f(x) = max( Z[true] - max_{i≠true} Z[i],  -kappa )
        # where Z are the raw logits (before softmax).
        #
        # Intuition:
        #   - Z[true] - max(Z[other]) is NEGATIVE when the attack succeeds
        #     (because some other class beats the true class)
        #   - We want this to be < -kappa, so we penalise when it's > -kappa
        #   - When the attack already succeeds with margin kappa, f = -kappa
        #     and adding more attack pressure only increases ||delta||, so
        #     Adam naturally stops perturbing further.

        logits = model(w)                          # [1, num_classes]
        true_logit = logits[0, true_index]

        # max over all classes EXCEPT the true class
        other_logits = logits.clone()
        other_logits[0, true_index] = -float("inf")
        best_other_logit = other_logits.max(dim=1).values[0]

        f_loss = torch.clamp(true_logit - best_other_logit, min=-kappa)

        # ── Total loss ────────────────────────────────────────────────
        loss = l2_loss + c * f_loss
        loss.backward()
        optimiser.step()

        # ── Track best adversarial example found so far ───────────────
        # "Best" = fooled the model AND has smallest L2 distance
        with torch.no_grad():
            pred = model(w).argmax(dim=1).item()
            current_l2 = l2_loss.item() ** 0.5     # sqrt to get L2 norm

            if pred != true_index and current_l2 < best_l2:
                best_l2   = current_l2
                best_adv  = w.detach().clone()
                best_pred = pred

    return best_adv, best_pred


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE WRAPPERS  (mirror the FGSM run_fgsm_pipeline interface)
# ─────────────────────────────────────────────────────────────────────────────

def run_cw_pipeline(model, device, filename, c_value, preprocess,
                    label_fn=None, **cw_kwargs):
    """
    Full C&W pipeline for ImageNet images.

    Args:
        c_value    : the c hyperparameter (replaces epsilon in FGSM)
        cw_kwargs  : optional overrides for kappa, lr, num_steps

    Returns:
        (pred_after, perturbed_tensor)
    """
    label_fn = label_fn or extract_true_label

    input_batch = get_input_batch(device, filename, preprocess)
    true_index, _ = label_fn(filename)

    perturbed, pred_after = cw_l2_attack(
        model, input_batch, true_index, device, c=c_value, **cw_kwargs
    )
    return pred_after, perturbed


def run_cw_pipeline_cifar(model, device, filename, c_value, preprocess,
                           **cw_kwargs):
    """
    Full C&W pipeline for CIFAR-10 images.
    Delegates to run_cw_pipeline with the CIFAR label extractor injected.
    """
    return run_cw_pipeline(
        model, device, filename, c_value, preprocess,
        label_fn=extract_true_label_cifar,
        **cw_kwargs
    )


# ─────────────────────────────────────────────────────────────────────────────
# ViT-SPECIFIC VARIANTS  (for HuggingFace ViTForImageClassification)
# ─────────────────────────────────────────────────────────────────────────────
#
# Why ViT needs its own functions:
#   torchvision models return a raw tensor of shape [batch, num_classes].
#   HuggingFace ViTForImageClassification returns a ModelOutput object
#   with a .logits attribute — model(x).logits is the tensor you want.
#   If you call model(x).argmax() directly it crashes because you can't
#   call argmax() on a ModelOutput object.
#
# Rule: always check what a model's forward() returns before calling
# methods on it. Torchvision → tensor. HuggingFace → ModelOutput.

def output_prediction_vit(model, input_batch):
    """
    Prediction for HuggingFace ViTForImageClassification.
    Extracts .logits before argmax, unlike the standard output_prediction().
    """
    with torch.no_grad():
        outputs = model(input_batch)
    return outputs.logits.argmax(dim=1).item()


def cw_l2_attack_vit(model, input_batch, true_index, device,
                     c=1e-2, kappa=0.0, lr=1e-2, num_steps=100):
    """
    C&W L2 attack for HuggingFace ViTForImageClassification.
    Identical logic to cw_l2_attack() but extracts .logits from model output
    before computing the attack loss — required for HuggingFace models.
    """
    x0 = input_batch.detach().clone()
    w  = x0.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([w], lr=lr)

    best_adv  = x0.clone()
    best_l2   = float("inf")
    best_pred = output_prediction_vit(model, x0)

    for step in range(num_steps):
        optimiser.zero_grad()

        l2_loss = torch.sum((w - x0) ** 2)

        # HuggingFace models return a ModelOutput — must use .logits
        logits = model(w).logits                   # [1, num_classes]
        true_logit = logits[0, true_index]

        other_logits = logits.clone()
        other_logits[0, true_index] = -float("inf")
        best_other_logit = other_logits.max(dim=1).values[0]

        f_loss = torch.clamp(true_logit - best_other_logit, min=-kappa)

        loss = l2_loss + c * f_loss
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            pred = model(w).logits.argmax(dim=1).item()
            current_l2 = l2_loss.item() ** 0.5

            if pred != true_index and current_l2 < best_l2:
                best_l2   = current_l2
                best_adv  = w.detach().clone()
                best_pred = pred

    return best_adv, best_pred


def run_cw_pipeline_cifar_vit(model, device, filename, c_value, preprocess,
                               **cw_kwargs):
    """
    Full C&W pipeline for CIFAR-10 images with HuggingFace ViT.
    Uses extract_true_label_cifar for labels and cw_l2_attack_vit for the attack.
    """
    input_batch = get_input_batch(device, filename, preprocess)
    true_index, _ = extract_true_label_cifar(filename)

    perturbed, pred_after = cw_l2_attack_vit(
        model, input_batch, true_index, device, c=c_value, **cw_kwargs
    )
    return pred_after, perturbed