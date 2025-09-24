from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def estimate_k_silhouette(X: np.ndarray, k_min=2, k_max=100, step=2, random_state=0):
    best_k, best_score = None, -1.0
    for k in range(k_min, k_max + 1, step):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric="cosine")
        if score > best_score:
            best_score, best_k = score, k
            best_centroids = km.cluster_centers_
    return best_k, best_centroids
