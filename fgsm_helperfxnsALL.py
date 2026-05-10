# based on fgsm_helperfxnsOG.py, but modified for ViT (e.g., different normalization values, different get_input_batch fxn)
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import json
import uuid
import os

# global counters
count_correct_beforeFGSM = 0
count_correct_afterFGSM = 0

# Constants
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

# get all image paths
def get_all_image_paths(root_dir):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(subdir, file))
    return image_paths
    

def get_input_batch(device, filename, preprocess):
    input_image = Image.open(filename).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    return input_batch



def output_prediction(model, input_batch):
    with torch.no_grad():
        output = model(input_batch)
    probabilities = F.softmax(output[0], dim=0)
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    return top1_catid.item()  # returning index of class



def extract_true_label(filename):
    # extract the full category folder name (eg "n06874185_traffic_light")
    category_folder = os.path.basename(os.path.dirname(filename))
    human_label = category_folder.split("_", 1)[-1]  # Get full label after first underscore

    with open("imagenet_class_index.json") as f:
        idx_to_label = json.load(f)

    for key, (wnid, label) in idx_to_label.items():
        if label == human_label:
            return int(key), label

    raise ValueError(f"Label '{human_label}' not found in class index.")



def compare_labels(predicted, true):
    return predicted == true



def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # perturbed_image = torch.clamp(perturbed_image, 0, 1) # TRIAL 1: remove claming to 
    return perturbed_image



def save_adv_image(img_tensor, epsilon, true_label, true_key, pred_before, pred_after,
                   output_dir="adv_outputs",
                   mean=None, std=None):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Inverse normalization
        inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )
        img_tensor = inv_normalize(img_tensor.squeeze().cpu())

        # Clamp pixel values to [0, 1]
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Convert to PIL image
        img_pil = transforms.ToPILImage()(img_tensor)

        # Generate unique filename
        uuid_suffix = uuid.uuid4().hex[:8]
        filename = f"{epsilon}_{true_label}{true_key}_init{pred_before}_adv{pred_after}_{uuid_suffix}.png"

        # Save
        save_path = os.path.join(output_dir, filename)
        img_pil.save(save_path)

        print(f"✅ Saved adversarial image: {save_path}")

    except Exception as e:
        print(f"❌ Failed to save adversarial image at eps={epsilon}: {e}")
        raise  # Ensure the exception propagates if needed



def run_fgsm_pipeline(model, device, filename, epsilon, preprocess): #removed default value for epsilon here so that we have to explicitly pass it in every time, which will help with counting correct per epsilon
    input_batch = get_input_batch(device, filename, preprocess)
    input_batch.requires_grad = True

    true_index, _ = extract_true_label(filename)

    # Forward pass
    output = model(input_batch)
    loss = F.nll_loss(F.log_softmax(output, dim=1), torch.tensor([true_index]).to(device))
    model.zero_grad()
    loss.backward()

    # FGSM
    data_grad = input_batch.grad.data
    perturbed_data = fgsm_attack(input_batch, epsilon, data_grad)

    # Predict on perturbed image
    output_adv = model(perturbed_data)
    pred_after = output_adv.argmax(dim=1).item() # this gives 

    return pred_after, perturbed_data # pred_


def extract_true_label_from_path(image_path: str):
    import re, os

    basename = os.path.basename(image_path)

    # Pattern: 0.1_zucchini939_init939_adv61_fd155200.png
    m = re.search(r'^[^_]+_([A-Za-z]+)(\d+)_init', basename)
    if m:
        label = m.group(1)
        index = int(m.group(2))
        return label, index

    # If pattern not found, fallback
    return "Unknown", -1


def extract_true_label_cifar(filename):
    # folder name IS the label, e.g. "cifar10_jpegs/test/airplane/00003_airplane.jpeg"
    category_folder = os.path.basename(os.path.dirname(filename))  # → "airplane"
    
    if category_folder not in CIFAR10_CLASSES:
        raise ValueError(f"Folder name '{category_folder}' is not a valid CIFAR-10 class.")
    
    index = CIFAR10_CLASSES.index(category_folder)  # → 0
    return index, category_folder


def run_fgsm_pipeline_cifar(model, device, filename, epsilon, preprocess): #removed default value for epsilon here so that we have to explicitly pass it in every time, which will help with counting correct per epsilon
    input_batch = get_input_batch(device, filename, preprocess)
    input_batch.requires_grad = True

    true_index, _ = extract_true_label_cifar(filename)

    # Forward pass
    output = model(input_batch)
    loss = F.nll_loss(F.log_softmax(output, dim=1), torch.tensor([true_index]).to(device))
    model.zero_grad()
    loss.backward()

    # FGSM
    data_grad = input_batch.grad.data
    perturbed_data = fgsm_attack(input_batch, epsilon, data_grad)

    # Predict on perturbed image
    output_adv = model(perturbed_data)
    pred_after = output_adv.argmax(dim=1).item() # this gives 

    return pred_after, perturbed_data # pred_