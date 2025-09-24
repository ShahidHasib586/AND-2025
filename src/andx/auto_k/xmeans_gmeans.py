from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def _bic_kmeans(X, labels, centroids):
    # crude BIC proxy for k-means
    n, d = X.shape
    k = centroids.shape[0]
    # within-cluster variance
    wcss = 0.0
    for j in range(k):
        Cj = X[labels == j]
        if len(Cj) == 0:
            continue
        wcss += ((Cj - centroids[j]) ** 2).sum()
    # parameters ~ k*d
    p = k * d
    return n * np.log(wcss / n + 1e-8) + p * np.log(n)

def estimate_k_xmeans_like(X: np.ndarray, k_min=2, k_max=100, random_state=0):
    best_k, best_bic = None, np.inf
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        bic = _bic_kmeans(X, labels, km.cluster_centers_)
        if bic < best_bic:
            best_bic, best_k = bic, k
            best_centroids = km.cluster_centers_
    return best_k, best_centroids

def estimate_k_gmeans_like(X: np.ndarray, k_min=1, k_max=100, random_state=0):
    # crude GMM-BIC sweep
    best_k, best_bic = None, np.inf
    for k in range(k_min, k_max + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic, best_k = bic, k
                best_centroids = gmm.means_
        except Exception:
            continue
    return best_k, best_centroids
