from __future__ import annotations
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Hungarian match
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return float(w[r, c].sum()) / y_pred.size

def compute_all(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"),
        "ARI": adjusted_rand_score(y_true, y_pred),
    }
