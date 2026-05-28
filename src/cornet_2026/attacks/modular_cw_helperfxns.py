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
    img = Image.open(filename).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    return tensor


def output_prediction(model, input_batch):
    with torch.no_grad():
        logits = model(input_batch)
    return logits.argmax(dim=1).item()


def compare_labels(predicted_index, true_index):
    return int(predicted_index) == int(true_index)


def extract_true_label(filename):
    category_folder = os.path.basename(os.path.dirname(filename))
    wnid = category_folder.split("_")[0]
    human_label = category_folder.split("_", 1)[-1]

    with open("imagenet_class_index.json") as f:
        idx_to_label = json.load(f)

    for idx, (synset, _) in idx_to_label.items():
        if synset == wnid:
            return int(idx), human_label

    raise ValueError(f"WNID {wnid} not found in imagenet_class_index.json")


def extract_true_label_cifar(filename):
    folder = os.path.basename(os.path.dirname(filename)).lower()
    if folder not in CIFAR10_CLASSES:
        raise ValueError(f"Folder '{folder}' is not a CIFAR-10 class.")
    return CIFAR10_CLASSES.index(folder), folder


def save_adv_image(img_tensor, c_value, true_label, true_key,
                   pred_before, pred_after,
                   output_dir="adv_outputs_cw",
                   mean=None, std=None):
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


def cw_l2_attack(model, input_batch, true_index, device,
                 c=1e-2, kappa=0.0, lr=1e-2, num_steps=100):
    x0 = input_batch.detach().clone()
    w = x0.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([w], lr=lr)

    best_adv   = x0.clone()
    best_l2    = float("inf")
    best_pred  = output_prediction(model, x0)

    for step in range(num_steps):
        optimiser.zero_grad()
        l2_loss = torch.sum((w - x0) ** 2)

        logits = model(w)
        true_logit = logits[0, true_index]

        other_logits = logits.clone()
        other_logits[0, true_index] = -float("inf")
        best_other_logit = other_logits.max(dim=1).values[0]

        f_loss = torch.clamp(true_logit - best_other_logit, min=-kappa)

        loss = l2_loss + c * f_loss
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            pred = model(w).argmax(dim=1).item()
            current_l2 = l2_loss.item() ** 0.5

            if pred != true_index and current_l2 < best_l2:
                best_l2   = current_l2
                best_adv  = w.detach().clone()
                best_pred = pred

    return best_adv, best_pred


def run_cw_pipeline(model, device, filename, c_value, preprocess,
                    label_fn=None, **cw_kwargs):
    label_fn = label_fn or extract_true_label

    input_batch = get_input_batch(device, filename, preprocess)
    true_index, _ = label_fn(filename)

    perturbed, pred_after = cw_l2_attack(
        model, input_batch, true_index, device, c=c_value, **cw_kwargs
    )
    return pred_after, perturbed


def run_cw_pipeline_cifar(model, device, filename, c_value, preprocess,
                           **cw_kwargs):
    return run_cw_pipeline(
        model, device, filename, c_value, preprocess,
        label_fn=extract_true_label_cifar,
        **cw_kwargs
    )


def output_prediction_vit(model, input_batch):
    with torch.no_grad():
        outputs = model(input_batch)
    return outputs.logits.argmax(dim=1).item()


def cw_l2_attack_vit(model, input_batch, true_index, device,
                     c=1e-2, kappa=0.0, lr=1e-2, num_steps=100):
    x0 = input_batch.detach().clone()
    w  = x0.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([w], lr=lr)

    best_adv  = x0.clone()
    best_l2   = float("inf")
    best_pred = output_prediction_vit(model, x0)

    for step in range(num_steps):
        optimiser.zero_grad()

        l2_loss = torch.sum((w - x0) ** 2)
        logits = model(w).logits
        true_logit = logits[0, true_index]

        other_logits = logits.clone()
        other_logits[0, true_index] = -float("inf")
        best_other_logit = other_logits.max(dim=1).values[0]

        f_loss = torch.clamp(true_logit - best_other_logit, min=-kappa)

        loss = l2_loss + c * f_loss
        loss.backward()
        optimiser.step()

    return best_adv, best_pred
