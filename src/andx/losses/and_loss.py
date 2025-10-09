from __future__ import annotations

import torch


class ANDLoss(torch.nn.Module):
    """Multi-positive InfoNCE with optional momentum feature bank."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        neighbor_idx: torch.Tensor,
        memory: "FeatureMemory" | None = None,
        sample_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        anchor = embeddings
        if memory is None or sample_indices is None:
            return self._batch_loss(anchor, neighbor_idx)
        bank = memory.all_features().to(anchor.device)
        valid_mask = memory.valid_mask().to(anchor.device)
        logits = anchor @ bank.t() / self.t
        if (~valid_mask).any():
            logits[:, ~valid_mask] = float("-inf")
        rows = torch.arange(anchor.size(0), device=anchor.device)
        logits[rows, sample_indices.to(anchor.device)] = float("-inf")
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pos_log_prob = log_prob.gather(1, neighbor_idx)
        loss = -pos_log_prob.mean(dim=1)
        return loss.mean()

    def _batch_loss(self, anchor: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
        logits = anchor @ anchor.t() / self.t
        mask = torch.eye(anchor.size(0), dtype=torch.bool, device=anchor.device)
        logits = logits.masked_fill(mask, float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pos_log_prob = log_prob.gather(1, neighbor_idx)
        loss = -pos_log_prob.mean(dim=1)
        return loss.mean()
