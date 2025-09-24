#!/usr/bin/env bash
set -e
# ResNet18
python -m src.hydra_main mode=train train=and dataset=cifar10 seed=42 train.model.backbone=resnet18
# ViT (timm)
python -m src.hydra_main mode=train train=and dataset=cifar10 seed=42 train.model.backbone="timm:vit_base_patch16_224"
# ConvNeXt tiny
python -m src.hydra_main mode=train train=and dataset=cifar10 seed=42 train.model.backbone="timm:convnext_tiny"
