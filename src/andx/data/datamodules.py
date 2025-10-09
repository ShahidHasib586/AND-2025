from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .transforms import build_transforms


class _IndexedDataset(Dataset):
    """Wrap a dataset to also return the sample index."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:  # pragma: no cover - thin wrapper
        return len(self.base)

    def __getitem__(self, index):  # pragma: no cover - delegation
        x, y = self.base[index]
        return x, y, index


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


class _BaseModule:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        mean = cfg.normalize["mean"]
        std = cfg.normalize["std"]
        self.t_train, self.t_test = build_transforms(cfg.img_size, mean, std, cfg.augment)
        self.ds_train = None
        self.ds_test = None

    def loaders(self, with_index: bool = False) -> Tuple[DataLoader, DataLoader]:
        train_ds = _IndexedDataset(self.ds_train) if with_index else self.ds_train
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
        )
        test_loader = DataLoader(
            self.ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        return train_loader, test_loader

    @property
    def num_train_samples(self) -> int:
        return len(self.ds_train)


class CIFAR10Module(_BaseModule):
    def __init__(self, cfg: DataConfig):
        super().__init__(cfg)
        self.ds_train = datasets.CIFAR10(cfg.root, train=True, transform=self.t_train, download=cfg.download)
        self.ds_test = datasets.CIFAR10(cfg.root, train=False, transform=self.t_test, download=cfg.download)


class STL10Module(_BaseModule):
    def __init__(self, cfg: DataConfig):
        super().__init__(cfg)
        self.ds_train = datasets.STL10(cfg.root, split="train", transform=self.t_train, download=cfg.download)
        self.ds_test = datasets.STL10(cfg.root, split="test", transform=self.t_test, download=cfg.download)


class TinyImageNetModule(_BaseModule):
    """Tiny-ImageNet expects ImageFolder layout under root/train and root/val."""

    def __init__(self, cfg: DataConfig):
        super().__init__(cfg)
        self.ds_train = datasets.ImageFolder(root=f"{cfg.root}/train", transform=self.t_train)
        self.ds_test = datasets.ImageFolder(root=f"{cfg.root}/val", transform=self.t_test)


def build_datamodule(cfg: DataConfig) -> _BaseModule:
    name = cfg.name.lower()
    if name == "cifar10":
        return CIFAR10Module(cfg)
    if name == "stl10":
        return STL10Module(cfg)
    if name == "tinyimagenet":
        return TinyImageNetModule(cfg)
    raise NotImplementedError(f"Dataset {cfg.name} not implemented")
