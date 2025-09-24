from __future__ import annotations
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

def linear_probe(train_feats, train_labels, test_feats, test_labels, epochs=50, lr=0.1, num_classes=10):
    clf = nn.Linear(train_feats.size(1), num_classes)
    opt = optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    ds_tr = TensorDataset(train_feats, train_labels)
    dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True)
    for _ in range(epochs):
        clf.train()
        for xb, yb in dl_tr:
            opt.zero_grad(set_to_none=True)
            logits = clf(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
    clf.eval()
    acc = (clf(test_feats).argmax(1) == test_labels).float().mean().item()
    return acc
