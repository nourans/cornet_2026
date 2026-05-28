"""
modular_fgsm_apply.py  –  unified fgsm attack script.

mirrors the structure of modular_cw_apply.py exactly.
the only things that change vs. the cw script:
  1. import from fgsm_helperfxnsALL instead of modular_cw_helperfxns
  2. the attack hyperparameter is `epsilon`, not `c`
  3. run_fgsm_pipeline* replace run_cw_pipeline*
  4. output directories are prefixed "adv_fgsm_outputs"

usage:
    python3 modular_fgsm_apply.py --model vit     --dataset cifar
    python3 modular_fgsm_apply.py --model resnet  --dataset imagenet
    python3 modular_fgsm_apply.py --model cornet  --dataset cifar --epsilons 0.005 0.01 0.1

with nohup:
    nohup python3 modular_fgsm_apply.py --model vit --dataset cifar \
        > logs/fgsm_vit_cifar.log 2>&1 &
"""

import argparse
import torch
import torchvision.transforms as transforms

from src.cornet_2026.attacks.fgsm_helperfxnsALL import (
    get_all_image_paths,
    get_input_batch,
    output_prediction,
    output_prediction_vit,
    compare_labels,
    save_adv_image,
    extract_true_label,
    extract_true_label_cifar,
    run_fgsm_pipeline,
    run_fgsm_pipeline_cifar,
    run_fgsm_pipeline_cifar_vit,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VIT_MEAN      = [0.5, 0.5, 0.5]
VIT_STD       = [0.5, 0.5, 0.5]


# ─────────────────────────────────────────────────────────────────────────────
# 1. model registry
# ─────────────────────────────────────────────────────────────────────────────

def build_model_config(model_name: str, dataset_name: str, device: torch.device) -> dict:
    if model_name == "vit":
        if dataset_name == "cifar":
            from transformers import ViTForImageClassification
            model = ViTForImageClassification.from_pretrained(
                "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
            ).to(device).eval()
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            mean, std       = IMAGENET_MEAN, IMAGENET_STD
            predict_fn      = output_prediction_vit
            run_pipeline_fn = run_fgsm_pipeline_cifar_vit
        else:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights    = ViT_B_16_Weights.IMAGENET1K_V1
            model      = vit_b_16(weights=weights).to(device).eval()
            preprocess = weights.transforms()
            mean, std       = VIT_MEAN, VIT_STD
            predict_fn      = output_prediction
            run_pipeline_fn = run_fgsm_pipeline

    elif model_name == "resnet":
        if dataset_name == "cifar":
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
            ).to(device).eval()
            preprocess = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            mean, std       = IMAGENET_MEAN, IMAGENET_STD
            predict_fn      = output_prediction
            run_pipeline_fn = run_fgsm_pipeline_cifar
        else:
            from torchvision.models import resnet50, ResNet50_Weights
            weights    = ResNet50_Weights.IMAGENET1K_V1
            model      = resnet50(weights=weights).to(device).eval()
            preprocess = weights.transforms()
            mean, std       = IMAGENET_MEAN, IMAGENET_STD
            predict_fn      = output_prediction
            run_pipeline_fn = run_fgsm_pipeline

    elif model_name == "cornet":
        from src.cornet_2026.models import cornet

        CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
        CIFAR_STD  = [0.2470, 0.2435, 0.2616]
        CKPT_PATH  = "./cornet_cifar10_best.pth"

        model = cornet.cornet_s(pretrained=False)
        model = model.module if isinstance(model, torch.nn.DataParallel) else model[0]
        model.decoder.linear = torch.nn.Linear(512, 10)

        checkpoint = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()

        if dataset_name == "cifar":
            preprocess = transforms.Compose([
                transforms.Resize(96),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
            ])
            run_pipeline_fn = run_fgsm_pipeline_cifar
        else:
            raise ValueError("this cornet checkpoint was trained on cifar-10 only. use --dataset cifar.")

        mean, std  = CIFAR_MEAN, CIFAR_STD
        predict_fn = output_prediction

    else:
        raise ValueError(f"unknown model '{model_name}'. choose: vit, resnet, cornet")

    return {
        "model":        model,
        "preprocess":   preprocess,
        "mean":         mean,
        "std":          std,
        "predict_fn":   predict_fn,
        "run_pipeline": run_pipeline_fn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. dataset registry
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_config(dataset_name: str) -> dict:
    if dataset_name == "cifar":
        return {
            "root_dir":      "cifar10_jpegs/test",
            "extract_label": extract_true_label_cifar,
        }
    elif dataset_name == "imagenet":
        return {
            "root_dir":      "val",
            "extract_label": extract_true_label,
        }
    else:
        raise ValueError(f"unknown dataset '{dataset_name}'. choose: cifar, imagenet")


# ─────────────────────────────────────────────────────────────────────────────
# 3. main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(model_name: str, dataset_name: str, epsilons: list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model_cfg   = build_model_config(model_name, dataset_name, device)
    dataset_cfg = build_dataset_config(dataset_name)

    model        = model_cfg["model"]
    preprocess   = model_cfg["preprocess"]
    mean         = model_cfg["mean"]
    std          = model_cfg["std"]
    predict_fn   = model_cfg["predict_fn"]
    run_pipeline = model_cfg["run_pipeline"]

    root_dir      = dataset_cfg["root_dir"]
    extract_label = dataset_cfg["extract_label"]

    def output_dir(eps):
        return f"adv_fgsm_outputs/{dataset_name}_{model_name}_eps{eps}"

    all_images            = get_all_image_paths(root_dir)
    total_images          = 0
    correct_before        = 0
    correct_after_per_eps = {eps: 0 for eps in epsilons}

    print(f"\n{'='*60}")
    print(f"fgsm attack | model: {model_name.upper()} | dataset: {dataset_name.upper()}")
    print(f"epsilons: {epsilons}")
    print(f"{'='*60}\n")

    for filename in all_images:
        try:
            total_images += 1

            try:
                input_batch = get_input_batch(device, filename, preprocess)
            except Exception as e:
                print(f"💔 can't load {filename}: {type(e).__name__}: {e}")
                continue

            true_index, true_label = extract_label(filename)
            print(f"true label: {true_label} (idx {true_index})")

            pred_before       = predict_fn(model, input_batch)
            is_correct_before = compare_labels(pred_before, true_index)
            if is_correct_before:
                correct_before += 1

            print(f"📊 predicted: {pred_before} | true: {true_index} | correct before: {is_correct_before}")

            for eps in epsilons:
                pred_after, perturbed_image = run_pipeline(
                    model, device, filename, eps, preprocess
                )

                try:
                    save_adv_image(
                        perturbed_image, eps, true_label, true_index,
                        pred_before, pred_after,
                        output_dir=output_dir(eps),
                        mean=mean, std=std,
                    )
                except Exception as e:
                    print(f"❌ save failed for {filename} at eps={eps}: {e}")

                is_correct_after = compare_labels(pred_after, true_index)
                if is_correct_after:
                    correct_after_per_eps[eps] += 1

                print(
                    f"{filename} | eps={eps} | "
                    f"correct_before={is_correct_before} | correct_after={is_correct_after}"
                )

            print(f"{'─'*60} {filename} done {'─'*60}")

        except Exception as e:
            print(f"error processing {filename}: {e}")

    print(f"\n{'='*60}")
    print(f"model: {model_name.upper()} | dataset: {dataset_name.upper()}")
    if total_images > 0:
        print(f"correct before fgsm: {correct_before}/{total_images} = {correct_before/total_images:.2%}")
        print(f"correct after fgsm (lower = stronger attack):")
        for eps in epsilons:
            acc = correct_after_per_eps[eps] / total_images
            print(f"  eps={eps}: {correct_after_per_eps[eps]}/{total_images} = {acc:.2%}")
    else:
        print("no images processed.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. cli entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run fgsm attack pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 modular_fgsm_apply.py --model vit    --dataset cifar
  python3 modular_fgsm_apply.py --model resnet --dataset imagenet --epsilons 0.005 0.01 0.1
  python3 modular_fgsm_apply.py --model cornet --dataset cifar --epsilons 0.01 0.1
        """
    )
    parser.add_argument("--model",    required=True, choices=["vit", "resnet", "cornet"])
    parser.add_argument("--dataset",  required=True, choices=["cifar", "imagenet"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.005, 0.01, 0.1])
    args = parser.parse_args()

    run(
        model_name   = args.model,
        dataset_name = args.dataset,
        epsilons     = args.epsilons,
    )