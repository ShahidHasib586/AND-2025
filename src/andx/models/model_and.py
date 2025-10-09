from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.resnet import ResNet18
from .backbones.vision_backbones import DINOv2Backbone, MAEBackbone, TimmBackbone
from ..losses.and_loss import ANDLoss


@dataclass
class ModelCfg:
    backbone: str = "resnet18"
    low_dim: int = 128
    projector: Dict = None
    autok_head: Dict = None  # handled in trainer if enabled


def build_model(cfg: ModelCfg):
    bb = cfg.backbone.lower()
    if bb == "resnet18":
        proj_dims = cfg.projector.get("dims", [512, 512, 128]) if cfg.projector else [512, 512, 128]
        return ResNet18(low_dim=cfg.low_dim, projector_dims=proj_dims)
    if bb.startswith("timm:"):
        name = bb.split("timm:", 1)[1]
        return TimmBackbone(name=name, pretrained=True, trainable=False)
    if bb.startswith("dinov2"):
        return DINOv2Backbone(name=cfg.backbone, trainable=False)
    if bb.startswith("mae"):
        return MAEBackbone(name=cfg.backbone, trainable=False)
    raise NotImplementedError(f"Backbone {cfg.backbone} not implemented")


class ANDCriterion(nn.Module):
    def __init__(self, temperature: float = 0.1, num_neighbors: int = 5):
        super().__init__()
        self.loss = ANDLoss(temperature)
        self.k = num_neighbors
        self.last_used_memory: bool = False
        self.last_neighbor_idx: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _knn_neighbors(self, z: torch.Tensor, k: int) -> torch.Tensor:
        sim = z @ z.t()
        idx = sim.topk(k + 1, dim=1).indices[:, 1:]
        return idx

    def forward(
        self,
        z: torch.Tensor,
        memory: Optional["FeatureMemory"] = None,
        sample_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = F.normalize(z, dim=1)
        neighbor_idx = None
        used_memory = False
        if memory is not None and sample_indices is not None:
            neighbor_idx = memory.query(z.detach(), sample_indices, self.k)
            used_memory = neighbor_idx is not None
        if neighbor_idx is None:
            neighbor_idx = self._knn_neighbors(z, self.k)
            sample_indices = None
            memory = None
        loss = self.loss(z, neighbor_idx, memory=memory, sample_indices=sample_indices)
        self.last_used_memory = used_memory
        self.last_neighbor_idx = neighbor_idx
        return loss
