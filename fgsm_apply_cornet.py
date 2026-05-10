"""
nohup python3 fgsm_apply_cornet.py > "*term_output_cornet_fgsm_cifar_0509_2108.txt" 2>&1 &
nohup python3 fgsm_apply_cornet.py > "*term_output_cornet_fgsm_imagenet_0509_2127.txt" 2>&1 &
"""
from cornet import cornet_s
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import uuid
import json
import os
from fgsm_helperfxnsCorRes import (
    get_all_image_paths, get_input_batch, output_prediction, extract_true_label,
    compare_labels, fgsm_attack, save_adv_image, run_fgsm_pipeline, extract_true_label_cifar
)

root_dir = "val" #imagenet100
#root_dir = "cifar10_jpegs/test" # cifar-10
all_images = get_all_image_paths(root_dir)

# Constants
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
epsilons = [0.005]#, 0.01, 0.1]
correct_before = 0
total_images = 0
# TRIAL 2: counting correct per epsilon
correct_after_per_eps = {eps: 0 for eps in epsilons}
total_per_eps = {eps: 0 for eps in epsilons}

# Load the model once globally
# Load pre-trained model on ImageNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cornet_s(pretrained=True, map_location=device)  # CORnet-S model with pretrained weights
model = model.module if hasattr(model, 'module') else model
model.eval()
model.to(device)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Loop through all images
for filename in all_images:
    try:
        total_images += 1
        try:
            input_batch = get_input_batch(device, filename, preprocess)
        except Exception as e:
            print(f"💔 can't do input_batch for {filename}: {type(e).__name__}: {e}")
        # Get true label: 
        # extract_true_label_cifar when working with CIFAR-10, extract_true_label when working with ImageNet
        true_index, true_label = extract_true_label(filename) 
        print(f"True label: {true_label}, index: {true_index}")

        # Get prediction before FGSM (this returns a string label)
        pred_before = output_prediction(model, input_batch)

        print(f"📊 Model predicted: {pred_before}, True index: {true_index}")

        # Compare string vs string (use true_label, not true_index)
        is_correct_before = compare_labels(pred_before, true_index) # changed this from true_label because pred_before is an index, not a string label. compare_labels should handle this correctly.
        if is_correct_before:
            correct_before += 1

        for eps in epsilons:
            # … run_fgsm_pipeline returns an int index …
            # pred_after = run_fgsm_pipeline(model, device, filename, eps)
            # added jun 10: start
            # Load AlexNet (once globally above or here if dynamic)
            # alexnet = models.alexnet(pretrained=True).to(device)
            # alexnet.eval()

            # Use CORnet to generate perturbed image
            pred_after, perturbed_image = run_fgsm_pipeline(model, device, filename, eps, preprocess)
            try:
                save_adv_image(
                perturbed_image, eps, true_label, true_index, pred_before, pred_after,
                output_dir=f"adv_CORoutputs1/adv_CORoutputs1_eps{eps}",
                mean=imagenet_mean, std=imagenet_std
            )
            # total_per_eps[eps] += 1 # commented out for new accuracy calc
            except Exception as e:
                print(f"❌ Failed to save image for {filename} at eps={eps}: {e}")

            # Compare indices (both ints)
            is_correct_after = compare_labels(pred_after, true_index) 
            # Add debugging output:
            print(f"{filename} | eps={eps} | correct before? {is_correct_before} | correct after? {is_correct_after}")

            if is_correct_after:
                correct_after_per_eps[eps] += 1
            
            # print(f"Correct for epsilon = {eps} is {correct_after}") # TRIAL 2: delete this

            # save_adv_image(
            #     perturbed_image, eps, true_label, true_index, pred_before, pred_after_cornet,
            #     output_dir=f"adv_outputs13/adv_outputs13_eps{eps}"
            # )
            # total_per_eps[eps] += 1
        print(f"---------------------------------------------------------{filename} ends---------------------------------------------------------")
    except Exception as e:
        print(f"Error processing {filename}: {e}")


# print(f"\nCorrect before FGSM: {correct_before}/{total_images} = {correct_before / total_images:.2%}")
# for eps in epsilons:
#     acc = correct_after_per_eps[eps] / total_per_eps[eps]
#     print(f"Epsilon {eps}: Accuracy after FGSM = {correct_after_per_eps[eps]}/{total_per_eps[eps]} = {acc:.2%}")

print(f"\nCorrect before FGSM: {correct_before}/{total_images} = {correct_before / total_images:.2%}")
for eps in epsilons:
    # if-else block commented out bc of new accuracy calculation method using total_images instead of total_per_eps[eps]
    # if total_per_eps[eps] == 0: 
    #     print(f"Epsilon {eps}: ⚠️ No adversarial images saved or processed.")

        # might wanna use current_count instead of total_per_eps[eps] for accuracy calculations.
        # it works now, but may cause problems in the future
        # acc = correct_after_per_eps[eps] / total_per_eps[eps]
        # print(f"Epsilon {eps}: Accuracy after FGSM = {correct_after_per_eps[eps]}/{total_per_eps[eps]} = {acc:.2%}")
    acc = correct_after_per_eps[eps] / total_images # total_per_eps[eps]
        # print(f"Epsilon {eps}: Accuracy after FGSM = {correct_after_per_eps[eps]}/{total_per_eps[eps]} = {acc:.2%}")
    print(f"Epsilon {eps}: Accuracy after FGSM = {correct_after_per_eps[eps]}/{total_images} = {acc:.2%}")
        