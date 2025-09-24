from __future__ import annotations
import torch
import torch.nn.functional as F

@torch.no_grad()
def knn_classifier(feat_train, y_train, feat_test, k=200, T=0.1):
    feat_train = F.normalize(feat_train, dim=1)
    feat_test = F.normalize(feat_test, dim=1)
    sim = feat_test @ feat_train.t() / T
    topk = sim.topk(k, dim=1).indices
    preds = []
    for i in range(topk.size(0)):
        neigh = y_train[topk[i]]
        # majority vote
        vals, counts = neigh.unique(return_counts=True)
        preds.append(vals[counts.argmax()])
    return torch.stack(preds)
