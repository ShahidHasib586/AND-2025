from __future__ import annotations
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian

def estimate_k_eigengap(X: np.ndarray, k_min=2, k_max=100, knn=15):
    # build affinity with cosine neighbors
    A = kneighbors_graph(X, n_neighbors=knn, metric="cosine", mode="connectivity")
    L = laplacian(A, normed=True)
    # eigenvalues of Laplacian
    w, _ = np.linalg.eigh(L.toarray())
    # eigengap heuristic: look for largest gap between consecutive eigenvalues
    gaps = np.diff(w[: min(len(w), k_max + 5)])  # slack
    idx = np.argmax(gaps[k_min - 1 : k_max - 1]) + (k_min - 1)
    k_est = idx + 1
    k_est = int(np.clip(k_est, k_min, k_max))
    # we dont compute centroids here, defer to k-means with k_est:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k_est, n_init="auto", random_state=0)
    labels = km.fit_predict(X)
    return k_est, km.cluster_centers_
