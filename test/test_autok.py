import torch, numpy as np
from src.andx.models.heads.autok_head import AutoKHead, AutoKCfg

def test_autok_head():
    head = AutoKHead(128, AutoKCfg(enabled=True))
    head.set_centroids(torch.randn(5,128))
    z = torch.randn(32,128)
    loss = head(z)
    assert loss.item() >= 0.0
