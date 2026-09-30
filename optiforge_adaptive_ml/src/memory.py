"""
Bounded replay memory for continual learning.

Internal storage is always <= K samples.  Incoming batches larger than K are
first reduced to a bounded candidate set, so replay compaction never creates an
internal historical array proportional to stream length.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union
import numpy as np

from src.task import TaskSpec


class StratifiedDensityReplayBuffer:
    """Target-aware, diversity-preserving replay buffer with a hard K-sample cap."""

    def __init__(
        self,
        task_spec: TaskSpec,
        max_samples: int = 500,
        n_strata: int = 10,
        random_seed: int = 42,
    ) -> None:
        self.task_spec = task_spec
        self.max_samples = max(1, int(max_samples))
        self.n_strata = max(2, int(n_strata))
        self.rng = np.random.default_rng(random_seed)

        self._X_buffer: np.ndarray | None = None
        self._y_buffer: np.ndarray | None = None
        self._strata_buffer: np.ndarray | None = None

        self.total_samples_observed = 0
        self.update_count = 0

    @property
    def current_size(self) -> int:
        return 0 if self._X_buffer is None else len(self._X_buffer)

    def get_memory_bytes(self) -> int:
        return sum(
            arr.nbytes
            for arr in (self._X_buffer, self._y_buffer, self._strata_buffer)
            if arr is not None
        )

    def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        return {
            "current_samples": self.current_size,
            "max_samples_capacity": self.max_samples,
            "capacity_utilization_pct": round(100.0 * self.current_size / self.max_samples, 2),
            "memory_bytes": self.get_memory_bytes(),
            "total_observed_lifetime": self.total_samples_observed,
            "update_count": self.update_count,
        }

    def _assign_strata(self, y: np.ndarray) -> np.ndarray:
        if self.task_spec.task_type == "binary_classification":
            return (np.asarray(y) > 0.5).astype(np.int8)
        if len(y) == 0:
            return np.empty(0, dtype=np.int16)
        edges = np.quantile(y, np.linspace(0.0, 1.0, self.n_strata + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return np.zeros(len(y), dtype=np.int16)
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        return np.clip(np.digitize(y, edges) - 1, 0, self.n_strata - 1).astype(np.int16)

    def _farthest_point_sampling(self, X: np.ndarray, quota: int) -> np.ndarray:
        n = len(X)
        if n <= quota:
            return np.arange(n)
        centroid = np.mean(X, axis=0, keepdims=True)
        selected = [int(np.argmin(np.sum((X - centroid) ** 2, axis=1)))]
        min_dist = np.sum((X - X[selected[0]]) ** 2, axis=1)
        for _ in range(1, quota):
            nxt = int(np.argmax(min_dist))
            selected.append(nxt)
            min_dist = np.minimum(min_dist, np.sum((X - X[nxt]) ** 2, axis=1))
        return np.asarray(selected, dtype=int)

    def _bounded_candidate_subset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce a large incoming batch to <= K candidates before combining."""
        if len(X) <= self.max_samples:
            return X, y

        strata = self._assign_strata(y)
        active = np.unique(strata)
        quotas = np.full(len(active), self.max_samples // max(1, len(active)), dtype=int)
        quotas[: self.max_samples % max(1, len(active))] += 1

        selected: List[int] = []
        for s, quota in zip(active, quotas):
            idx = np.flatnonzero(strata == s)
            if len(idx) <= quota:
                selected.extend(idx.tolist())
            else:
                local = self._farthest_point_sampling(X[idx], int(quota))
                selected.extend(idx[local].tolist())

        selected = selected[: self.max_samples]
        return X[selected], y[selected]

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> "StratifiedDensityReplayBuffer":
        X_arr = np.asarray(X_new, dtype=np.float32)
        y_arr = np.asarray(y_new, dtype=float)
        if X_arr.ndim != 2 or len(X_arr) != len(y_arr):
            raise ValueError("X_new and y_new must have compatible shapes")

        self.total_samples_observed += len(X_arr)
        self.update_count += 1
        if len(X_arr) == 0:
            return self

        X_candidate, y_candidate = self._bounded_candidate_subset(X_arr, y_arr)

        if self._X_buffer is None:
            combined_X, combined_y = X_candidate, y_candidate
        else:
            combined_X = np.vstack((self._X_buffer, X_candidate))
            combined_y = np.concatenate((self._y_buffer, y_candidate))

        if len(combined_X) <= self.max_samples:
            selected = np.arange(len(combined_X))
        else:
            strata = self._assign_strata(combined_y)
            active = np.unique(strata)
            quotas = np.full(len(active), self.max_samples // len(active), dtype=int)
            quotas[: self.max_samples % len(active)] += 1
            selected_list: List[int] = []

            for s, quota in zip(active, quotas):
                idx = np.flatnonzero(strata == s)
                if len(idx) <= quota:
                    selected_list.extend(idx.tolist())
                else:
                    local = self._farthest_point_sampling(combined_X[idx], int(quota))
                    selected_list.extend(idx[local].tolist())

            selected = np.asarray(selected_list, dtype=int)
            if len(selected) > self.max_samples:
                selected = self.rng.choice(selected, self.max_samples, replace=False)

        self._X_buffer = np.ascontiguousarray(combined_X[selected].copy())
        self._y_buffer = np.ascontiguousarray(combined_y[selected].copy())
        self._strata_buffer = self._assign_strata(self._y_buffer)
        assert self.current_size <= self.max_samples
        return self

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._X_buffer is None or len(self._X_buffer) == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=float)
        return self._X_buffer.copy(), self._y_buffer.copy()
