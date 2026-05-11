"""
nohup python3 fgsm_apply_resnet.py > "*term_output_fgsm_imgnt_resnet_0509_2314.txt" 2>&1 &
nohup python3 fgsm_apply_resnet.py > "*term_output_fgsm_cifar_resnet_0509_2337.txt" 2>&1 &


when switching between CIFAR-10 and ImageNet, remember to change:
- root_dir (line 30/31)  
- epsilon values (line 36)
- preprocessing for CIFAR-10 vs ImageNet (line 53)
- extract_true_label function (line 71)
- run_fgsm_pipeline (line 93)
- the file we save to in save_adv_image (line 97)
"""
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import uuid
import json
import os
from fgsm_helperfxnsALL import (
    get_all_image_paths, get_input_batch, output_prediction, compare_labels, fgsm_attack, save_adv_image, 
    extract_true_label, extract_true_label_cifar, 
    run_fgsm_pipeline, run_fgsm_pipeline_cifar
)

# root_dir = "val" #imagenet100
root_dir = "cifar10_jpegs/test" # cifar-10
all_images = get_all_image_paths(root_dir)

# Constants
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

epsilons = [0.005, 0.01, 0.1]
correct_before = 0
total_images = 0
# TRIAL 2: counting correct per epsilon
correct_after_per_eps = {eps: 0 for eps in epsilons}
total_per_eps = {eps: 0 for eps in epsilons}

# Load the model once globally
# Load pre-trained model on ImageNet
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.eval()
# preprocess = weights.transforms() # IMAGENET: Preprocess and classify

# cifar-10
preprocess = transforms.Compose([
   transforms.Resize(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
assert torch.cuda.is_available(), "CUDA is not available — check your GPU setup"

model.to(device)

# Loop through all images
for filename in all_images:
    try:
        total_images += 1
        try:
            input_batch = get_input_batch(device, filename, preprocess)
        except Exception as e:
            print(f"💔 can't do input_batch for {filename}: {type(e).__name__}: {e}")
        # Get true label
        # extract_true_label_cifar when working with CIFAR-10, extract_true_label when working with ImageNet
        true_index, true_label = extract_true_label_cifar(filename) 
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
            pred_after, perturbed_image = run_fgsm_pipeline_cifar(model, device, filename, eps, preprocess)
            try:
                save_adv_image(
                    perturbed_image, eps, true_label, true_index, pred_before, pred_after,
                    output_dir=f"adv_cifar_RESoutputs1/adv_cifar_RESoutputs1_eps{eps}",
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
        