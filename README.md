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
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1 #. .venv/Scripts/activate
#                                 # Windows Git Bash/CMD: .venv\Scripts\activate

# 3. Install the package in editable mode (ensures `src/` is importable)
python -m pip install --upgrade pip
python -m pip install -e .       # alternatively: python -m pip install -r requirements.txt

# 4. (Optional) Developer tooling
python -m pip install pre-commit
pre-commit install
```

> **Troubleshooting:** If you see `ModuleNotFoundError: No module named 'src.andx.utils.memory'`, make sure the virtual
> environment is activated and re-run `python -m pip install -e .`. The editable install wires the `src/` tree as a Python
> package so the training entry point can resolve intra-project imports.

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

#### Switching Auto-K estimators
Each estimator exposes different override knobs. The table below lists the flag to set on the command line (via Hydra) and
notable parameters you may want to tweak:

| Method name        | Override example                                                                 | Notes |
|--------------------|-----------------------------------------------------------------------------------|-------|
| `consensus`        | `train.model.autok_head.method=consensus train.model.autok_head.sweep.k_max=80`   | Runs multiple k-means sweeps and aggregates votes. Use `train.model.autok_head.sweep.*` to control the k range. |
| `silhouette`       | `train.model.autok_head.method=silhouette train.model.autok_head.sweep.step=1`    | Maximises the cosine silhouette score of k-means assignments. |
| `eigengap`         | `train.model.autok_head.method=eigengap train.model.autok_head.sweep.k_min=2`     | Selects k based on the spectral eigengap heuristic. |
| `dpmeans`          | `train.model.autok_head.method=dpmeans`                                          | Chooses k via DP-means with the default lambda (tweak inside the source if needed). |
| `xmeans` (G-means) | `train.model.autok_head.method=xmeans train.model.autok_head.sweep.k_max=128`     | Runs an X-means/G-means style growth heuristic over the specified sweep range. |

All Auto-K heads share warm-up and update parameters that can be overridden as well, e.g. `train.model.autok_head.update_freq=2`
or `train.model.autok_head.warmup_epochs=5`.

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

### Visualising clustering progress
1. **Monitor training live with TensorBoard**:
   ```bash
   tensorboard --logdir runs
   ```
   Watch the `autok/k` scalar (current estimate of the number of clusters), the `autok/q` quality proxy, and the feature histograms written at the end of every training round.

2. **Inspect embeddings offline** once a checkpoint is saved:
   ```bash
   # Dump features for the evaluation split via the utility helper
   python - <<'PY'
   from omegaconf import OmegaConf
   from src.build_features import dump_features

   dataset_cfg = OmegaConf.load('configs/dataset/cifar10.yaml')
   dump_features(
       'runs/train/checkpoint_and_or_autok.pt',
       'outputs/cifar10_features.pt',
       dataset_cfg,
   )
   PY
   ```
   The resulting tensor file (`Z` for embeddings, `y` for labels) can be loaded in `noteboooks/viz_embeddings.ipynb` to project features with UMAP/t-SNE and visually inspect cluster separation. Launch Jupyter with `jupyter notebook` or open the notebook directly in VS Code.

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
