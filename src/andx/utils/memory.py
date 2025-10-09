from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class MemoryStats:
    valid: int
    capacity: int


class FeatureMemory:
    """Momentum feature bank used to mine neighbours across minibatches."""

    def __init__(self, num_samples: int, dim: int, momentum: float = 0.5, device: Optional[torch.device] = None):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum should be in [0, 1]")
        self.num_samples = int(num_samples)
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.device = device or torch.device("cpu")

        self._features = torch.zeros(self.num_samples, self.dim, device=self.device)
        self._valid = torch.zeros(self.num_samples, dtype=torch.bool, device=self.device)

    def to(self, device: torch.device) -> "FeatureMemory":
        self.device = device
        self._features = self._features.to(device)
        self._valid = self._valid.to(device)
        return self

    @property
    def stats(self) -> MemoryStats:
        return MemoryStats(valid=int(self._valid.sum().item()), capacity=self.num_samples)

    def update(self, indices: torch.Tensor, features: torch.Tensor) -> None:
        if indices.ndim != 1:
            raise ValueError("indices must be 1D")
        if features.shape[0] != indices.shape[0]:
            raise ValueError("features and indices must align on batch dimension")
        indices = indices.to(self.device, non_blocking=True)
        feats = F.normalize(features.detach().to(self.device), dim=1)

        existing_mask = self._valid[indices]
        if existing_mask.any():
            idx_existing = indices[existing_mask]
            updated = self.momentum * self._features[idx_existing] + (1.0 - self.momentum) * feats[existing_mask]
            self._features[idx_existing] = F.normalize(updated, dim=1)

        if (~existing_mask).any():
            idx_new = indices[~existing_mask]
            self._features[idx_new] = feats[~existing_mask]
            self._valid[idx_new] = True

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        return self._features[indices.to(self.device)]

    def query(self, queries: torch.Tensor, sample_indices: torch.Tensor, k: int) -> Optional[torch.Tensor]:
        if self._valid.sum() < k:
            return None
        queries = F.normalize(queries, dim=1)
        sims = queries @ self._features.t()
        invalid = ~self._valid.to(queries.device)
        if invalid.any():
            sims[:, invalid] = float("-inf")
        row = torch.arange(sample_indices.shape[0], device=queries.device)
        sims[row, sample_indices.to(queries.device)] = float("-inf")
        topk = torch.topk(sims, k, dim=1).indices
        return topk

    def all_features(self) -> torch.Tensor:
        return self._features

    def valid_mask(self) -> torch.Tensor:
        return self._valid
