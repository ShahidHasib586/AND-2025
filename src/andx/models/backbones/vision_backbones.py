from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn

def _freeze(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m

class TimmBackbone(nn.Module):
    """
    Wraps a timm model to emit a pooled feature vector.
    Example names: 'resnet50', 'vit_base_patch16_224', 'convnext_tiny'
    """
    def __init__(self, name: str = "vit_base_patch16_224", pretrained: bool = True, trainable: bool = False):
        super().__init__()
        import timm  # lazy import
        self.model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.feat_dim = self.model.num_features
        if not trainable:
            _freeze(self.model)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.model(x)          # (B, D)
        return f, f                # projector can be Identity if using frozen feats

class DINOv2Backbone(nn.Module):
    """
    DINOv2 from torch.hub (FacebookResearch/dinov2).
    If not available, raise a clear error.
    """
    def __init__(self, name: str = "dinov2_vitb14", trainable: bool = False):
        super().__init__()
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", name)
        except Exception as e:
            raise RuntimeError("Install DINOv2 weights or use TimmBackbone instead.") from e
        self.feat_dim = self.model.embed_dim
        if not trainable:
            _freeze(self.model)

    def forward(self, x):
        f = self.model(x)  # returns tokens pooled (B, D)
        return f, f

class MAEBackbone(nn.Module):
    """
     MAE (ViT) via timm pretrains: e.g. 'vit_base_patch16_224.mae'
    """
    def __init__(self, name: str = "vit_base_patch16_224.mae", trainable: bool = False):
        super().__init__()
        import timm
        self.model = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
        self.feat_dim = self.model.num_features
        if not trainable:
            _freeze(self.model)

    def forward(self, x):
        f = self.model(x)
        return f, f
