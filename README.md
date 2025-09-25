# Unsupervised deep clustering by neighbourhood discovery with automatic estimation of the number of clusters.

Modern PyTorch 2.x reimplementation of **AND** with an **Auto-K** clustering head that estimates the number of clusters during training. Reproducible, Hydra-configurable, and packaged for 2025 toolchains.

## 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
# or: pip install -r requirements.txt
pre-commit install
```
## 2. Dataset

By default CIFAR-10 is downloaded to ./data/cifar10. Can be override via config or CLI:

dataset.root=~/data/cifar10


## Train (AND only)
```bash
python -m src.hydra_main mode=train train=and dataset=cifar10

```
## Train (AND + Auto-K)
```bash
python -m src.hydra_main mode=train train=and_autok dataset=cifar10

```

## Evaluate
```bash
python -m src.hydra_main mode=eval +ckpt=path/to/checkpoint.pt


```

Useful CLI overrides

seed=123

device=cpu

dataset.batch_size=256

train.max_epochs=100

train.model.backbone=resnet18

## 6. Repro

We ship configs to mirror original hyperparams (SGD 0.03 lr, step decay, etc.). Set seed and log dirs to reproduce.



---

### `scripts/train_and.sh`
```bash
#!/usr/bin/env bash
set -e
python -m src.hydra_main mode=train train=and dataset=cifar10 seed=42
```
### scripts/train_and_autok.sh
```bash
#!/usr/bin/env bash
set -e
python -m src.hydra_main mode=train train=and_autok dataset=cifar10 seed=42
```
### scripts/eval.sh
```bash
#!/usr/bin/env bash
set -e
CKPT=${1:-runs/train/checkpoint_and_only.pt}
python -m src.hydra_main mode=eval +ckpt=$CKPT
```

## 7. How to run (quick)

```bash

# 0) clone your new repo
git clone https://github.com/ShahidHasib586/AND-2025.git
cd AND-2025
```

```bash
# 1) env
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

```bash
# 2) train (AND)
python -m src.hydra_main mode=train train=and dataset=cifar10
```

```bash
# 3) train AND with Auto-K, consensus, default sweep
python -m src.hydra_main mode=train train=and_autok dataset=cifar10 seed=42

```

```bash
# 4) eval
python -m src.hydra_main mode=eval +ckpt=runs/train/checkpoint_and_only.pt

```
```bash

# 5) Evaluate a checkpoint

python -m src.hydra_main mode=eval +ckpt=runs/train/checkpoint_and_or_autok.pt


```

overrides

choose Auto-K method:

train=and_autok train.model.autok_head.method=silhouette


change sweep:

train=and_autok train.model.autok_head.sweep.k_min=2 train.model.autok_head.sweep.k_max=50 train.model.autok_head.sweep.step=1


force CPU:

device=cpu
