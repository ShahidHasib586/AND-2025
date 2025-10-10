# AND-2025

Unsupervised deep clustering by neighbourhood discovery (AND) with automatic estimation of the number of clusters (**Auto-K**), reimplemented on top of the modern PyTorch 2.x toolchain. The project wraps the original algorithm in a reproducible, Hydra-driven training pipeline and exposes multiple Auto-K strategies so that the number of clusters can be discovered during training instead of being fixed a priori.

## Key features
- **PyTorch 2.x implementation** with optional `torch.compile` acceleration and mixed precision support.
- **Hydra configuration system**: every experiment is driven by YAML configs under [`configs/`](configs/) and can be customised from the command line.
- **Automatic K estimation** via consensus, silhouette, eigengap, DP-means, and X-means/G-means inspired heuristics.
- **Reproducible experiments** thanks to deterministic seeding, versioned configs, and scriptable entry points.
- **Logging out of the box** to TensorBoard (default) with optional Weights & Biases integration.
- **Packaging & testing ready**: installable with `pip install -e .`, `pytest` test suite, and `pre-commit` hooks for formatting.

## Requirements
- Python **3.10** or newer
- PyTorch **2.2** + CUDA 11.8 (or CPU-only build)
- (Optional) NVIDIA GPU with at least 8 GB VRAM for CIFAR-10 scale experiments

All runtime dependencies are declared in [`pyproject.toml`](pyproject.toml). Development extras such as formatting hooks live in [`.pre-commit-config.yaml`](.pre-commit-config.yaml) (install with `pre-commit install`).

## Installation
```bash
# 1. Clone the repository
git clone https://github.com/ShahidHasib586/AND-2025.git
cd AND-2025

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the package
pip install --upgrade pip
pip install -e .                 # or: pip install -r requirements.txt

# 4. (Optional) Install git hooks
pre-commit install
```

## Datasets
The default experiments target **CIFAR-10**. When first launched, the dataset will be downloaded to `./data/cifar10`. Override the root path via Hydra:

```bash
python -m src.hydra_main dataset.root=~/datasets/cifar10
```

Custom datasets can be added by extending the configs under [`configs/dataset/`](configs/dataset/).

## Running experiments
The single entry point is [`src/hydra_main.py`](src/hydra_main.py). Select the mode (`train` or `eval`) and stack overrides as needed.

### Train AND (baseline)
```bash
python -m src.hydra_main mode=train train=and dataset=cifar10
```

### Train AND with Auto-K
```bash
python -m src.hydra_main \
    mode=train \
    train=and_autok \
    dataset=cifar10 \
    seed=42
```

Auto-K configuration lives inside [`configs/train/and_autok.yaml`](configs/train/and_autok.yaml). For instance, choose the estimator and sweep range with:

```bash
python -m src.hydra_main \
    mode=train \
    train=and_autok \
    train.model.autok_head.method=silhouette \
    train.model.autok_head.sweep.k_min=2 \
    train.model.autok_head.sweep.k_max=50 \
    train.model.autok_head.sweep.step=1
```

### Evaluate checkpoints
```bash
python -m src.hydra_main mode=eval +ckpt=runs/train/checkpoint_and_only.pt
```
Override the checkpoint location with `+ckpt=/path/to/model.pt`.

### Common overrides
| Purpose                     | Example override(s) |
|----------------------------|---------------------|
| Run on CPU                 | `device=cpu`
| Change batch size          | `dataset.batch_size=256`
| Limit epochs per round     | `train.max_epochs=100`
| Switch backbone encoder    | `train.model.backbone=resnet18`
| Change RNG seed            | `seed=123`

## Convenience scripts
Shortcuts for the most common workflows live under [`scripts/`](scripts/):

- [`scripts/train_and.sh`](scripts/train_and.sh)
- [`scripts/train_and_autok.sh`](scripts/train_and_autok.sh)
- [`scripts/eval.sh`](scripts/eval.sh)

Each script simply wraps the corresponding `python -m src.hydra_main ...` command with sensible defaults.

## Outputs & logging
Runs are stored in `./runs/` by default. TensorBoard logs are written to `runs/*/events.out.tfevents...`. Enable Weights & Biases logging by setting `log.wandb=true` (and providing authentication via `WANDB_API_KEY`).

## Testing
Execute the unit tests and style checks locally before submitting changes:

```bash
pytest
pre-commit run --all-files
```

## Project structure
```
├── configs/            # Hydra configuration tree (dataset, training, logging)
├── docs/               # MkDocs documentation sources
├── scripts/            # Helper shell scripts for training/evaluation
├── src/                # AND models, Auto-K heads, training & evaluation code
├── test/               # Pytest-based regression tests
└── outputs/, runs/     # Experiment artefacts (created at runtime)
```

## Citing
If you build on this implementation, please cite the original AND paper as well as this repository.
