import torch
import torch.nn.functional as F

class ANDLoss(torch.nn.Module):
    """
    Minimalized proxy of AND objective:
    - Anchor neighbourhood consistency via soft nearest positive/negative mining
    - Temperature-scaled logits; momentum updates handled in trainer
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, z: torch.Tensor, neigh_idx: torch.Tensor):
        # z: (B, D) normalized embeddings
        # neigh_idx: (B, K) indices of neighbours within the current batch
        z = F.normalize(z, dim=1)
        B, D = z.shape
        K = neigh_idx.shape[1]
        anchor = z
        neigh = z[neigh_idx]           # (B, K, D)
        pos = neigh.mean(dim=1)         # simple aggregation of neighbours
        logits_pos = (anchor * pos).sum(-1) / self.t
        # negatives: all other samples in batch
        logits_all = (anchor @ z.t()) / self.t
        mask = torch.eye(B, device=z.device).bool()
        logits_all = logits_all.masked_fill(mask, -1e9)
        # InfoNCE-like
        loss = -logits_pos + torch.logsumexp(logits_all, dim=1)
        return loss.mean()
