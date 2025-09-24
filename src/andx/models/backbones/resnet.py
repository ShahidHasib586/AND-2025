from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as tv

class ResNet18(nn.Module):
    def __init__(self, low_dim: int = 128, projector_dims=(512,512,128)):
        super().__init__()
        base = tv.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])  # 512 x 1 x 1
        self.feat_dim = 512
        proj = []
        in_dim = self.feat_dim
        for d in projector_dims:
            proj += [nn.Linear(in_dim, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True)]
            in_dim = d
        self.projector = nn.Sequential(*proj[:-1])   # drop last ReLU for embedding
        self.head = nn.Identity()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x).flatten(1)      # (B, 512)
        z = self.projector(f)                # (B, low_dim)
        return f, z
