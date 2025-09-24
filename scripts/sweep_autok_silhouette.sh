#!/usr/bin/env bash
set -e
for KMAX in 20 50 100; do
  python -m src.hydra_main mode=train train=and_autok dataset=cifar10 seed=42 \
    train.model.autok_head.method=silhouette \
    train.model.autok_head.sweep.k_min=2 \
    train.model.autok_head.sweep.k_max=$KMAX \
    train.model.autok_head.sweep.step=1
done
