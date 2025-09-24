#!/usr/bin/env bash
set -e
for LAM in 0.5 1.0 2.0 4.0; do
  python -m src.hydra_main mode=train train=and_autok dataset=cifar10 seed=42 \
    train.model.autok_head.method=dpmeans +train.model.autok_head.lam=$LAM
done
