from dataclasses import dataclass
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from .transforms import build_transforms
from torchvision import datasets, transforms
@dataclass
class DataConfig:
    name: str
    root: str
    num_classes: int
    img_size: int
    normalize: dict
    augment: dict
    batch_size: int
    num_workers: int
    pin_memory: bool
    download: bool

class CIFAR10Module:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        mean = cfg.normalize["mean"]
        std  = cfg.normalize["std"]
        self.t_train, self.t_test = build_transforms(cfg.img_size, mean, std, cfg.augment)
        self.ds_train = datasets.CIFAR10(cfg.root, train=True,  transform=self.t_train, download=cfg.download)
        self.ds_test  = datasets.CIFAR10(cfg.root, train=False, transform=self.t_test,  download=cfg.download)

    def loaders(self) -> Tuple[DataLoader, DataLoader]:
        dl_train = DataLoader(self.ds_train, batch_size=self.cfg.batch_size, shuffle=True,
                              num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, drop_last=True)
        dl_test  = DataLoader(self.ds_test, batch_size=self.cfg.batch_size, shuffle=False,
                              num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory)
        return dl_train, dl_test

class STL10Module:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        mean = cfg.normalize["mean"]; std = cfg.normalize["std"]
        t_train, t_test = build_transforms(cfg.img_size, mean, std, cfg.augment)
        self.ds_train = datasets.STL10(cfg.root, split="train", transform=t_train, download=cfg.download)
        self.ds_test  = datasets.STL10(cfg.root, split="test",  transform=t_test,  download=cfg.download)
    def loaders(self):
        return (
            DataLoader(self.ds_train, batch_size=self.cfg.batch_size, shuffle=True,
                       num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, drop_last=True),
            DataLoader(self.ds_test,  batch_size=self.cfg.batch_size, shuffle=False,
                       num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory),
        )

class TinyImageNetModule:
    """
    Expects Tiny-ImageNet-200 in ImageFolder layout under root:
      root/train/*/images/*.JPEG
      root/val/*/images/*.JPEG  , convert val annotations to subfolders if needed
    """
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        mean = cfg.normalize["mean"]; std = cfg.normalize["std"]
        t_train, t_test = build_transforms(cfg.img_size, mean, std, cfg.augment)
        self.ds_train = datasets.ImageFolder(root=f"{cfg.root}/train", transform=t_train)
        self.ds_test  = datasets.ImageFolder(root=f"{cfg.root}/val",   transform=t_test)
    def loaders(self):
        return (
            DataLoader(self.ds_train, batch_size=self.cfg.batch_size, shuffle=True,
                       num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, drop_last=True),
            DataLoader(self.ds_test,  batch_size=self.cfg.batch_size, shuffle=False,
                       num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory),
        )

def build_datamodule(cfg: DataConfig):
    name = cfg.name.lower()
    if name == "cifar10":
        return CIFAR10Module(cfg)
    if name == "stl10":
        return STL10Module(cfg)
    if name == "tinyimagenet":
        return TinyImageNetModule(cfg)
    raise NotImplementedError(f"Dataset {cfg.name} not implemented")