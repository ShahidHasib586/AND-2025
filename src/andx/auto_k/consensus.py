from __future__ import annotations
import numpy as np
from .silhouette import estimate_k_silhouette
from .xmeans_gmeans import estimate_k_xmeans_like, estimate_k_gmeans_like
from .eigengap import estimate_k_eigengap

def estimate_k_consensus(X: np.ndarray, k_min=2, k_max=100, step=2, random_state=0):
    votes = []
    centroids_pool = []

    for fn in (
        lambda: estimate_k_silhouette(X, k_min, k_max, step, random_state),
        lambda: estimate_k_xmeans_like(X, k_min, k_max, random_state),
        lambda: estimate_k_gmeans_like(X, max(1, k_min-1), k_max, random_state),
        lambda: estimate_k_eigengap(X, k_min, k_max),
    ):
        try:
            k_hat, C = fn()
            votes.append(k_hat)
            centroids_pool.append((k_hat, C))
        except Exception:
            pass

    if not votes:
        # fallback
        from sklearn.cluster import KMeans
        k_hat = max(2, min(10, k_max))
        km = KMeans(n_clusters=k_hat, n_init="auto", random_state=random_state).fit(X)
        return k_hat, km.cluster_centers_

    # median vote for robustness
    k_hat = int(np.median(votes))
    # pick centroids from the estimator with k closest to the median
    best = min(centroids_pool, key=lambda t: abs(t[0] - k_hat))
    return k_hat, best[1]
