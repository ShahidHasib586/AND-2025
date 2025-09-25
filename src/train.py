from __future__ import annotations
import os
from pathlib import Path

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from omegaconf import DictConfig, OmegaConf

from .andx.seed import set_seed
from .andx.logging import get_logger
from .andx.data.datamodules import DataConfig, build_datamodule
from .andx.models.model_and import ModelCfg, build_model, ANDCriterion

from .andx.models.heads.autok_head import AutoKHead, AutoKCfg
from .andx.auto_k.consensus import estimate_k_consensus
from .andx.auto_k.silhouette import estimate_k_silhouette
from .andx.auto_k.xmeans_gmeans import estimate_k_xmeans_like
from .andx.auto_k.eigengap import estimate_k_eigengap
from .andx.auto_k.dp_means import estimate_k_dpmeans
from .andx.auto_k.stability import KStability
from .andx.auto_k.stability import KStability
from .andx.auto_k.consensus import estimate_k_consensus
from .andx.auto_k.silhouette import estimate_k_silhouette
from .andx.auto_k.xmeans_gmeans import estimate_k_xmeans_like
from .andx.auto_k.eigengap import estimate_k_eigengap
from .andx.auto_k.dp_means import estimate_k_dpmeans

def make_optimizer(params, cfg):
    if cfg.name == "sgd":
        return optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        )
    raise NotImplementedError(f"Unsupported optimizer: {cfg.name}")


def train_one_epoch(model, criterion, loader, opt, device, console, step_base, writer, epoch, display_freq=100):
    import torch.nn.functional as F  # not strictly needed, but handy if we add extra logging

    model.train()
    running = 0.0
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        _, z = model(x)
        loss = criterion(z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += loss.item()
        global_step = step_base + i
        if writer:
            writer.add_scalar("train/loss", loss.item(), global_step)
        if (i + 1) % display_freq == 0:
            console.log(f"[epoch {epoch}] iter {i+1}/{len(loader)} loss {running/(i+1):.4f}")
    return running / max(len(loader), 1)


def validate_embeddings(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, z = model(x)
            feats.append(z.cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def _choose_autok_fn(name: str):
    return {
        "consensus": estimate_k_consensus,
        "silhouette": estimate_k_silhouette,
        "xmeans": estimate_k_xmeans_like,
        "eigengap": estimate_k_eigengap,
        "dpmeans": lambda X, **kw: estimate_k_dpmeans(X, lam=kw.get("lam", 1.0)),
    }[name]


def main_train(cfg: DictConfig):
    # ---- seed & device
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    # ---- logging
    out_dir = os.path.join(cfg.output_dir, "train")
    console, writer = get_logger(Path(out_dir), use_tb=cfg.log.tb)

    # ---- data
    dcfg = DataConfig(**cfg.dataset)
    dm = build_datamodule(dcfg)
    train_loader, test_loader = dm.loaders()

    # ---- model
    mcfg = ModelCfg(**cfg.train.model)
    model = build_model(mcfg).to(device)
    if cfg.compile and device.type == "cuda":
        model = torch.compile(model)

    # ---- losses / opt / sched
    crit_and = ANDCriterion(cfg.train.loss.npc_temperature).to(device)
    opt = make_optimizer(model.parameters(), cfg.train.optimizer)
    sched = StepLR(opt, step_size=cfg.train.lr_schedule.step, gamma=cfg.train.lr_schedule.gamma)

    # ---- optional Auto-K head
    autok_cfg_dict = cfg.train.model.get("autok_head", None)
    autok = None
    autok_fn = None
    kstab = None
    if autok_cfg_dict and autok_cfg_dict.get("enabled", False):
        autok_cfg = AutoKCfg(**autok_cfg_dict)
        autok = AutoKHead(feat_dim=mcfg.low_dim, cfg=autok_cfg).to(device)
        autok_fn = _choose_autok_fn(autok_cfg.method)
        kstab = KStability(window=3, min_delta_ratio=autok_cfg_dict.get("min_delta", 0.01))

    # ---- training (progressive rounds x epochs)
    global_step = 0
    for r in range(cfg.train.rounds):
        console.rule(f"Round {r}")
        for epoch in range(cfg.train.max_epochs):
            # one epoch of AND
            loss = train_one_epoch(
                model, crit_and, train_loader, opt, device, console, global_step, writer, epoch, cfg.train.display_freq
            )
            sched.step()
            global_step += len(train_loader)

            # periodic Auto-K update + regularizer (if enabled)
            if autok:
                # add clustering regularizer during training after warmup
                if epoch >= autok.cfg.warmup_epochs and autok.current_k > 1:
                    # quick extra pass over a small subset could be added if needed; we keep it simple
                    pass

                # re-estimate K on validation embeddings every update_freq epochs
                if epoch >= autok.cfg.warmup_epochs and (epoch % autok.cfg.update_freq == 0):
                    model.eval()
                    with torch.no_grad():
                        Z, _ = validate_embeddings(model, test_loader, device)  # (N, D)
                    X = Z.cpu().numpy()

                    # choose estimator
                    if autok.cfg.method == "dpmeans":
                        k_new, C = autok_fn(X, lam=1.0)
                        q = 1.0 / (k_new + 1e-6)  # crude quality proxy
                    elif autok.cfg.method in ("consensus", "silhouette"):
                        from sklearn.metrics import silhouette_score
                        from sklearn.cluster import KMeans

                        k_new, C = autok_fn(X, **autok.cfg.sweep or {"k_min": 2, "k_max": 100, "step": 2})
                        labels = KMeans(n_clusters=k_new, n_init="auto", random_state=0).fit_predict(X)
                        q = silhouette_score(X, labels, metric="cosine") if len(set(labels)) > 1 else -1.0
                    else:
                        # xmeans/eigengap -> quality proxy via inverse inertia
                        from sklearn.cluster import KMeans

                        k_new, C = autok_fn(X, **autok.cfg.sweep or {"k_min": 2, "k_max": 100, "step": 2})
                        km = KMeans(n_clusters=k_new, n_init="auto", random_state=0).fit(X)
                        q = 1.0 / (km.inertia_ / X.shape[0] + 1e-9)

                    accepted_k = kstab.propose(k_new, q) if kstab else k_new
                    if accepted_k != autok.current_k:
                        from sklearn.cluster import KMeans

                        C_acc = KMeans(n_clusters=accepted_k, n_init="auto", random_state=0).fit(X).cluster_centers_
                        autok.set_centroids(torch.from_numpy(C_acc).float().to(device))
                    if writer:
                        writer.add_scalar("autok/k", autok.current_k, r * cfg.train.max_epochs + epoch)
                        writer.add_scalar("autok/q", q, r * cfg.train.max_epochs + epoch)

        # feature histogram per round
        Z_round, _ = validate_embeddings(model, test_loader, device)
        if writer:
            writer.add_histogram("feats/std", Z_round.std(dim=0), r)

    # save
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "autok": (autok.state_dict() if autok else None),
        },
        os.path.join(out_dir, "checkpoint_and_or_autok.pt"),
    )
    console.log("Training complete.")
