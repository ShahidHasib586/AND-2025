from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import ResNet18
from ..losses.and_loss import ANDLoss

from .backbones.resnet import ResNet18
from .backbones.vision_backbones import TimmBackbone, DINOv2Backbone, MAEBackbone

@dataclass
class ModelCfg:
    backbone: str = "resnet18"
    low_dim: int = 128
    projector: Dict = None
    autok_head: Dict = None  # handled in trainer if enabled

def build_model(cfg: ModelCfg):
    bb = cfg.backbone.lower()
    if bb == "resnet18":
        proj_dims = cfg.projector.get("dims", [512,512,128]) if cfg.projector else [512,512,128]
        return ResNet18(low_dim=cfg.low_dim, projector_dims=proj_dims)
    if bb.startswith("timm:"):
        name = bb.split("timm:")[1]
        return TimmBackbone(name=name, pretrained=True, trainable=False)
    if bb.startswith("dinov2"):
        return DINOv2Backbone(name=cfg.backbone, trainable=False)
    if bb.startswith("mae"):
        return MAEBackbone(name=cfg.backbone, trainable=False)
    raise NotImplementedError(f"Backbone {cfg.backbone} not implemented")

class ANDCriterion(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.loss = ANDLoss(temperature)

    @torch.no_grad()
    def _knn_neighbors(self, z, k=5):
        z = F.normalize(z, dim=1)
        sim = z @ z.t()
        idx = sim.topk(k+1, dim=1).indices[:,1:]  # drop self
        return idx

    def forward(self, z):
        neigh = self._knn_neighbors(z, k=5)
        return self.loss(z, neigh)
