"""
cw_apply.py  –  Unified Carlini & Wagner (L2) attack script.

Mirrors the structure of fgsm_apply.py exactly.
The only things that change vs. the FGSM script:
  1. Import from cw_helperfxnsALL instead of fgsm_helperfxnsALL
  2. The attack hyperparameter is `c` (trade-off constant), not `epsilon`
  3. run_cw_pipeline / run_cw_pipeline_cifar replace run_fgsm_pipeline*
  4. Output directories are prefixed "adv_cw_outputs" for easy separation

Usage:
    python3 cw_apply.py --model vit     --dataset cifar
    python3 cw_apply.py --model resnet  --dataset imagenet
    python3 cw_apply.py --model cornet  --dataset cifar --c_values 0.001 0.01 0.1

With nohup:
    nohup python3 cw_apply.py --model vit --dataset cifar \
        > logs/cw_vit_cifar.log 2>&1 &

Choosing c_values:
    c controls the trade-off between perturbation size and attack success:
      small c (e.g. 1e-4) → minimal distortion, may not always fool the model
      large c (e.g. 1.0)  → strong attack, larger (but still L2-small) distortion
    Typical values used in the literature: [0.001, 0.01, 0.1]
    These are analogous (but NOT equivalent) to epsilon in FGSM.
"""

import argparse
import torch
import torchvision.transforms as transforms

from modular_cw_helperfxns import (
    get_all_image_paths,
    get_input_batch,
    output_prediction,
    compare_labels,
    save_adv_image,
    extract_true_label,
    extract_true_label_cifar,
    run_cw_pipeline,
    run_cw_pipeline_cifar,
    IMAGENET_MEAN, IMAGENET_STD,
    VIT_MEAN, VIT_STD,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL REGISTRY
#    Same pattern as fgsm_apply.py: vary what changes, freeze what doesn't.
#    To add a new model → add one elif here. The main loop never changes.
# ─────────────────────────────────────────────────────────────────────────────

def build_model_config(model_name: str, device: torch.device) -> dict:
    """Load model + preprocessing for the requested model_name."""

    if model_name == "vit":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights   = ViT_B_16_Weights.IMAGENET1K_V1
        model     = vit_b_16(weights=weights).to(device).eval()
        preprocess = weights.transforms()
        mean, std  = VIT_MEAN, VIT_STD

    elif model_name == "resnet":
        from torchvision.models import resnet50, ResNet50_Weights
        weights    = ResNet50_Weights.IMAGENET1K_V1
        model      = resnet50(weights=weights).to(device).eval()
        preprocess = weights.transforms()
        mean, std  = IMAGENET_MEAN, IMAGENET_STD

    elif model_name == "cornet":
        import cornet
        # CORnet loads wrapped in DataParallel; unwrap for single-device use.
        # Rule: DataParallel splits a batch across GPUs. On one GPU (or CPU)
        # you must call .module to access the underlying model directly.
        model = cornet.cornet_s(pretrained=True).module.to(device).eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose: vit, resnet, cornet")

    return {"model": model, "preprocess": preprocess, "mean": mean, "std": std}


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET REGISTRY
#    Injects the correct label extractor and pipeline function.
#    This is the Strategy Pattern: the loop calls extract_label() and
#    run_pipeline() without knowing which dataset is underneath.
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_config(dataset_name: str) -> dict:
    """Return root directory, label extractor, and pipeline function."""

    if dataset_name == "cifar":
        return {
            "root_dir":        "cifar10_jpegs/test",
            "extract_label":   extract_true_label_cifar,
            "run_pipeline":    run_cw_pipeline_cifar,
        }
    elif dataset_name == "imagenet":
        return {
            "root_dir":        "val",
            "extract_label":   extract_true_label,
            "run_pipeline":    run_cw_pipeline,
        }
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose: cifar, imagenet")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN LOOP
#    Identical structure to fgsm_apply.py.
#    The only difference: we iterate over c_values instead of epsilons,
#    and call run_pipeline with c=c_val instead of epsilon=eps.
# ─────────────────────────────────────────────────────────────────────────────

def run(model_name: str, dataset_name: str, c_values: list,
        cw_kappa: float = 0.0,
        cw_lr:    float = 1e-2,
        cw_steps: int   = 100):
    """
    Run the C&W attack pipeline.

    Args:
        model_name   : "vit", "resnet", or "cornet"
        dataset_name : "cifar" or "imagenet"
        c_values     : list of c trade-off constants to sweep
        cw_kappa     : confidence margin (default 0 = just flip prediction)
        cw_lr        : Adam learning rate inside the C&W optimiser
        cw_steps     : number of optimisation steps per image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model and dataset configs ───────────────────────────────────────
    model_cfg   = build_model_config(model_name, device)
    dataset_cfg = build_dataset_config(dataset_name)

    model      = model_cfg["model"]
    preprocess = model_cfg["preprocess"]
    mean       = model_cfg["mean"]
    std        = model_cfg["std"]

    root_dir      = dataset_cfg["root_dir"]
    extract_label = dataset_cfg["extract_label"]
    run_pipeline  = dataset_cfg["run_pipeline"]

    # ── Output directory factory ─────────────────────────────────────────────
    # Each (model, dataset, c) combination gets its own folder so parallel
    # runs never overwrite each other.
    def output_dir(c_val):
        return f"adv_cw_outputs/{dataset_name}_{model_name}_c{c_val}"

    # ── Counters ─────────────────────────────────────────────────────────────
    all_images = get_all_image_paths(root_dir)
    total_images = 0
    correct_before = 0
    correct_after_per_c = {c: 0 for c in c_values}

    print(f"\n{'='*60}")
    print(f"C&W Attack | Model: {model_name.upper()} | Dataset: {dataset_name.upper()}")
    print(f"c values: {c_values} | kappa={cw_kappa} | lr={cw_lr} | steps={cw_steps}")
    print(f"{'='*60}\n")

    # ── Main loop ────────────────────────────────────────────────────────────
    for filename in all_images:
        try:
            total_images += 1

            # ── Load image ───────────────────────────────────────────────────
            try:
                input_batch = get_input_batch(device, filename, preprocess)
            except Exception as e:
                print(f"💔 Can't load {filename}: {type(e).__name__}: {e}")
                continue  # Rule: always `continue` after catching load errors
                           # so that undefined `input_batch` never reaches
                           # the code below and causes a confusing NameError.

            # ── True label ───────────────────────────────────────────────────
            true_index, true_label = extract_label(filename)
            print(f"True label: {true_label} (idx {true_index})")

            # ── Prediction BEFORE attack ─────────────────────────────────────
            pred_before       = output_prediction(model, input_batch)
            is_correct_before = compare_labels(pred_before, true_index)
            if is_correct_before:
                correct_before += 1

            print(f"📊 Predicted: {pred_before} | True: {true_index} | Correct before: {is_correct_before}")

            # ── Sweep over c values (analogous to epsilon sweep in FGSM) ─────
            for c_val in c_values:
                pred_after, perturbed_image = run_pipeline(
                    model, device, filename, c_val, preprocess,
                    kappa=cw_kappa, lr=cw_lr, num_steps=cw_steps,
                )

                try:
                    save_adv_image(
                        perturbed_image, c_val, true_label, true_index,
                        pred_before, pred_after,
                        output_dir=output_dir(c_val),
                        mean=mean, std=std,
                    )
                except Exception as e:
                    print(f"❌ Save failed for {filename} at c={c_val}: {e}")

                is_correct_after = compare_labels(pred_after, true_index)
                print(
                    f"{filename} | c={c_val} | "
                    f"before={is_correct_before} | after={is_correct_after}"
                )

                if is_correct_after:
                    correct_after_per_c[c_val] += 1

            print(f"{'─'*60} {filename} done {'─'*60}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Model: {model_name.upper()} | Dataset: {dataset_name.upper()}")
    if total_images > 0:
        print(f"Correct BEFORE C&W: {correct_before}/{total_images} = {correct_before/total_images:.2%}")
        for c_val in c_values:
            acc = correct_after_per_c[c_val] / total_images
            print(f"  c={c_val}: {correct_after_per_c[c_val]}/{total_images} = {acc:.2%}")
    else:
        print("No images processed.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Carlini & Wagner L2 attack pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 cw_apply.py --model vit    --dataset cifar
  python3 cw_apply.py --model resnet --dataset imagenet --c_values 0.001 0.01 0.1
  python3 cw_apply.py --model cornet --dataset cifar --steps 200 --lr 0.005
        """
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["vit", "resnet", "cornet"],
        help="Which model to attack",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar", "imagenet"],
        help="Which dataset to use",
    )
    parser.add_argument(
        "--c_values",
        nargs="+",
        type=float,
        default=[0.01, 0.1],
        help="Space-separated c trade-off values (e.g. --c_values 0.001 0.01 0.1)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.0,
        help="Confidence margin (default 0 = just flip prediction)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Adam learning rate inside C&W optimiser (default 0.01)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of C&W optimisation steps per image (default 100)",
    )
    args = parser.parse_args()

    run(
        model_name   = args.model,
        dataset_name = args.dataset,
        c_values     = args.c_values,
        cw_kappa     = args.kappa,
        cw_lr        = args.lr,
        cw_steps     = args.steps,
    )