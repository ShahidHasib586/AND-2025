from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class AutoKCfg:
    enabled: bool = True
    method: str = "consensus"  # "silhouette" | "xmeans" | "eigengap" | "dpmeans" | "consensus"
    sweep: Dict = None         # {k_min, k_max, step}
    update_freq: int = 1
    min_delta: float = 0.01
    warmup_epochs: int = 10
    lambda_cluster: float = 1.0

class AutoKHead(nn.Module):
    """
    A lightweight clustering head that:
    1) keeps a small set of prototypes (centroids)
    2) periodically re-estimates K and re-initialises centroids
    3) provides a clustering regularizer loss during training
    """
    def __init__(self, feat_dim: int, cfg: AutoKCfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("centroids", torch.zeros(1, feat_dim))
        self.current_k: int = 1
        self.feat_dim = feat_dim

    @torch.no_grad()
    def set_centroids(self, centroids: torch.Tensor):
        self.current_k = centroids.shape[0]
        self.centroids = centroids.detach().to(self.centroids.device)

    def forward(self, z: torch.Tensor):
        # z: (B, D), centroids: (K, D)
        if self.current_k <= 1:
            return torch.tensor(0.0, device=z.device)
        z = F.normalize(z, dim=1)
        c = F.normalize(self.centroids, dim=1)
        logits = z @ c.t()  # cosine similarity
        # Regularizer: sharpened assignment with entropy minimisation
        p = F.softmax(logits / 0.1, dim=1)
        sharpen = (p ** 2) / (p ** 2).sum(dim=1, keepdim=True)
        q = sharpen.detach()
        loss = F.kl_div((p + 1e-6).log(), q, reduction="batchmean")
        return loss * self.cfg.lambda_cluster
