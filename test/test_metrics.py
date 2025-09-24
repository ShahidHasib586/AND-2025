import numpy as np
from src.andx.metrics.clustering_scores import compute_all

def test_metrics():
    y = np.array([0,0,1,1])
    yhat = np.array([1,1,0,0])
    m = compute_all(y, yhat)
    assert 0.0 <= m["ACC"] <= 1.0
