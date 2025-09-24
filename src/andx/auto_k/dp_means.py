from __future__ import annotations
import numpy as np

def dp_means(X: np.ndarray, lam: float, max_iter=100):
    n, d = X.shape
    centroids = [X[np.random.randint(0, n)]]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # assign
        C = np.vstack(centroids)
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        min_d2 = d2.min(axis=1)
        labels = d2.argmin(axis=1)
        # spawn new clusters
        for i in range(n):
            if min_d2[i] > lam:
                centroids.append(X[i])
                labels[i] = len(centroids) - 1
        # update
        C_new = []
        for k in range(len(centroids)):
            pts = X[labels == k]
            if len(pts) == 0:
                C_new.append(centroids[k])
            else:
                C_new.append(pts.mean(axis=0))
        if np.allclose(np.vstack(centroids), np.vstack(C_new)):
            break
        centroids = C_new
    return np.vstack(centroids), labels

def estimate_k_dpmeans(X: np.ndarray, lam: float = 1.0):
    C, labels = dp_means(X, lam=lam)
    return C.shape[0], C
