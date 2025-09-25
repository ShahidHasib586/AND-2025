from __future__ import annotations
import numpy as np
from collections import deque

class KStability:
    """
    Debounced acceptance of K updates based on a sliding window of (k, quality) pairs.
    Accepts a new K if the median K changes by at least min_delta_ratio
    AND the median quality improves versus the previous accepted quality.
    """
    def __init__(self, window=3, min_delta_ratio=0.05):
        self.window = int(window)
        self.buf = deque(maxlen=self.window)
        self.min_delta = float(min_delta_ratio)
        self.prev_k = None
        self.prev_score = None  # higher-is-better (e.g., silhouette or inverse inertia)

    def propose(self, k_new: int, quality_score: float):
        self.buf.append((int(k_new), float(quality_score)))
        if len(self.buf) < self.window:
            # not enough evidence yet; keep previous (or return the only k we have)
            return self.prev_k if self.prev_k is not None else k_new

        ks, qs = zip(*self.buf)
        k_med = int(np.median(ks))
        q_med = float(np.median(qs))

        if self.prev_k is None:
            self.prev_k, self.prev_score = k_med, q_med
            return k_med

        rel_change = abs(k_med - self.prev_k) / max(self.prev_k, 1)
        improved = (self.prev_score is None) or (q_med > self.prev_score)

        if rel_change >= self.min_delta and improved:
            self.prev_k, self.prev_score = k_med, q_med
            return k_med

        return self.prev_k
