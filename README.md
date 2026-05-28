# cornet_2026

A structured PyTorch repository for CORnet-based image classification and adversarial attack experiments.

## Repository layout

- `src/cornet_2026/`
  - `models/` — CORnet model definitions and entrypoint wrappers
  - `attacks/` — adversarial attack helper modules for FGSM and C&W
  - `data/` — dataset loader utilities and dataset-related helpers
  - `utils/` — shared utility modules
- `scripts/` — runnable scripts for training, evaluation, and attack generation
- `data/` — datasets and dataset assets used by the scripts
- `outputs/` — adversarial outputs, logs, and generated artifacts
- `vendor/CORnet/` — vendored external CORnet implementation
- `archive/` — historical experiments and legacy scripts
- `tests/` — evaluation and sanity-check scripts
- `requirements.txt` — Python dependencies

## Setup

1. Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Python from the repository root with `src` on `PYTHONPATH`:

```bash
PYTHONPATH=./src python3 scripts/train_cornet_cifar10.py
```

> If you want, a `pyproject.toml` or editable install can be added later to simplify this.

## Common workflows

### Training CORnet on CIFAR-10

```bash
PYTHONPATH=./src python3 scripts/train_cornet_cifar10.py
```

### Fine-tuning CORnet on CIFAR-10

```bash
PYTHONPATH=./src python3 scripts/finetune_cornet_cifar.py
```

### Running FGSM attacks

```bash
PYTHONPATH=./src python3 scripts/modular_fgsm_apply.py --model cornet --dataset cifar
```

### Running C&W attacks

```bash
PYTHONPATH=./src python3 scripts/modular_cw_apply.py --model resnet --dataset imagenet
```

## Notes

- `vendor/CORnet/` contains vendored CORnet model code that is kept separate from your package source.
- Keep scripts in `scripts/` and package source under `src/cornet_2026/` for a cleaner, conventional structure.
- Add dataset files to `data/` and generated outputs to `outputs/`.

## Recommended next steps

- Add `pyproject.toml` or `setup.cfg` to enable `pip install -e .`
- Move more helpers into `src/cornet_2026/data/` and `src/cornet_2026/utils/` as the project grows
- Add tests to `tests/` to verify training, model loading, and attack pipelines
