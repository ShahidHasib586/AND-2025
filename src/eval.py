from __future__ import annotations
import torch, numpy as np
from omegaconf import DictConfig
from .andx.data.datamodules import DataConfig, build_datamodule
from .andx.models.model_and import ModelCfg, build_model
from .andx.knn_eval.knn import knn_classifier
from .andx.knn_eval.linear_probe import linear_probe
from .andx.metrics.clustering_scores import compute_all
from sklearn.cluster import KMeans

@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        _, z = model(x)
        feats.append(z.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)

def main_eval(cfg: DictConfig):
    device = torch.device("cuda" if (cfg.device=="cuda" and torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    mcfg = ModelCfg(**ckpt["cfg"]["train"]["model"])
    model = build_model(mcfg).to(device)
    model.load_state_dict(ckpt["model"])

    dcfg = DataConfig(**cfg.dataset)
    dm = build_datamodule(dcfg)
    tr, te = dm.loaders()
    Ztr, Ytr = extract(model, tr, device)
    Zte, Yte = extract(model, te, device)

    # kNN
    pred_knn = knn_classifier(Ztr, Ytr, Zte, k=cfg.protocols[0]["knn"]["k"], T=cfg.protocols[0]["knn"]["T"])
    knn_acc = (pred_knn == Yte).float().mean().item()

    # linear probe
    lp_cfg = [p for p in cfg.protocols if "linear_probe" in p][0]["linear_probe"]
    lp_acc = linear_probe(Ztr, Ytr, Zte, Yte, epochs=lp_cfg["epochs"], lr=lp_cfg["lr"], num_classes=dcfg.num_classes)

    # clustering metrics (run k-means with GT k for reporting)
    km = KMeans(n_clusters=dcfg.num_classes, n_init="auto", random_state=0).fit(Zte.numpy())
    clus = compute_all(Yte.numpy(), km.labels_)

    print(f"kNN@{cfg.protocols[0]['knn']['k']} acc: {knn_acc:.4f}")
    print(f"Linear probe acc: {lp_acc:.4f}")
    print(f"Clustering (k-means, k={dcfg.num_classes}) ACC={clus['ACC']:.4f} NMI={clus['NMI']:.4f} ARI={clus['ARI']:.4f}")
