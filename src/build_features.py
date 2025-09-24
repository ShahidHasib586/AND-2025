from __future__ import annotations
import os, torch
from omegaconf import OmegaConf
from .andx.data.datamodules import DataConfig, build_datamodule
from .andx.models.model_and import ModelCfg, build_model

@torch.no_grad()
def dump_features(ckpt_path: str, out_path: str, dataset_cfg):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", None)
    model = build_model(ModelCfg(**cfg["train"]["model"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dcfg = DataConfig(**dataset_cfg)
    dm = build_datamodule(dcfg)
    _, test_loader = dm.loaders()

    feats, labels = [], []
    for x, y in test_loader:
        _, z = model(x)
        feats.append(z)
        labels.append(y)
    Z = torch.cat(feats).cpu()
    y = torch.cat(labels).cpu()
    torch.save({"Z": Z, "y": y}, out_path)
