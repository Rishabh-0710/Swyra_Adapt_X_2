"""
Bounded statistical distribution-shift detection for continual tabular learning.

The detector keeps a fixed-size reference reservoir rather than retaining the
entire initial dataset.  PSI and normalized Wasserstein compare each incoming
batch with that bounded reference; residual CUSUM is evaluated strictly from
pre-update predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import wasserstein_distance


@dataclass
class DriftReport:
    overall_drift_score: float
    severity: str
    is_concept_drift: bool
    concept_drift_score: float
    feature_psi: Dict[str, float] = field(default_factory=dict)
    feature_wasserstein: Dict[str, float] = field(default_factory=dict)
    top_drifted_features: List[str] = field(default_factory=list)
    confidence: float = 1.0
    covariate_drift_score: float = 0.0
    drift_type: str = "none"


class OnlineResidualCUSUM:
    """Bounded residual change detector using baseline moments only."""

    def __init__(self, threshold: float = 3.5, drift_delta: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.drift_delta = float(drift_delta)
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.n_samples = 0
        self.is_initialized = False

    def initialize_baseline(self, residuals: np.ndarray) -> None:
        res = np.asarray(residuals, dtype=float)
        res = res[np.isfinite(res)]
        self.baseline_mean = float(np.mean(res)) if res.size else 0.0
        self.baseline_std = float(max(1e-4, np.std(res))) if res.size else 1.0
        self.s_pos = self.s_neg = 0.0
        self.n_samples = int(res.size)
        self.is_initialized = True

    def update(self, residuals: np.ndarray) -> Tuple[bool, float]:
        if not self.is_initialized:
            return False, 0.0
        res = np.asarray(residuals, dtype=float)
        res = res[np.isfinite(res)]
        if res.size == 0:
            return False, 0.0

        z = (res - self.baseline_mean) / self.baseline_std
        mean_z = float(np.mean(z))
        self.s_pos = max(0.0, self.s_pos + mean_z - self.drift_delta)
        self.s_neg = min(0.0, self.s_neg + mean_z + self.drift_delta)

        cusum_stat = max(self.s_pos, abs(self.s_neg))
        # A batch mean is an estimate of the residual mean.  Scaling it by
        # sqrt(n) gives a sample-size-aware standardized mean statistic.
        batch_stat = abs(mean_z) * np.sqrt(max(1, len(res)))
        stat = max(cusum_stat, batch_stat)
        score = float(np.clip(stat / max(self.threshold, 1e-6), 0.0, 2.0))
        detected = bool(stat >= self.threshold)

        if detected:
            self.s_pos *= 0.5
            self.s_neg *= 0.5
        return detected, score


class DriftDetector:
    """
    Domain-agnostic detector with O(K*D) reference memory.

    ``reference_size`` is a hard cap independent of initial dataset size and
    stream length.  The reservoir is updated with classic streaming reservoir
    sampling, and all distribution summaries are rebuilt from that bounded set.
    """

    def __init__(
        self,
        psi_low: float = 0.10,
        psi_high: float = 0.25,
        wasserstein_threshold: float = 0.20,
        cusum_threshold: float = 3.5,
        n_bins: int = 10,
        reference_size: int = 256,
        min_samples_for_detection: int = 50,
        random_seed: int = 42,
    ) -> None:
        self.psi_low = float(psi_low)
        self.psi_high = float(psi_high)
        self.wasserstein_threshold = float(wasserstein_threshold)
        self.n_bins = max(4, int(n_bins))
        self.reference_size = max(32, int(reference_size))
        self.min_samples_for_detection = max(10, int(min_samples_for_detection))
        self.rng = np.random.default_rng(random_seed)

        self.reference_X: Optional[np.ndarray] = None
        self.reference_stds: Optional[np.ndarray] = None
        self.bin_edges: Dict[int, np.ndarray] = {}
        self.ref_bin_probs: Dict[int, np.ndarray] = {}
        self.reference_seen = 0
        self.cusum = OnlineResidualCUSUM(threshold=cusum_threshold)
        self.null_drift_floor: float = 0.0

    @property
    def reference_size_current(self) -> int:
        return 0 if self.reference_X is None else len(self.reference_X)

    def _rebuild_distribution_summary(self) -> None:
        if self.reference_X is None or len(self.reference_X) == 0:
            return

        X_arr = self.reference_X
        n_samples, n_features = X_arr.shape
        self.reference_stds = np.std(X_arr, axis=0)
        self.reference_stds[self.reference_stds < 1e-6] = 1.0
        self.bin_edges = {}
        self.ref_bin_probs = {}

        quantiles = np.linspace(0, 100, self.n_bins + 1)
        for j in range(n_features):
            vals = X_arr[:, j]
            edges = np.percentile(vals, quantiles)
            lo, hi = float(np.min(vals)), float(np.max(vals))
            if not np.isfinite(lo) or not np.isfinite(hi):
                edges = np.linspace(-1.0, 1.0, self.n_bins + 1)
            else:
                if hi <= lo:
                    edges = np.linspace(lo - 1.0, hi + 1.0, self.n_bins + 1)
                else:
                    edges[0] = min(edges[0], lo) - 1e-6
                    edges[-1] = max(edges[-1], hi) + 1e-6
                    edges = np.unique(edges)
                    if len(edges) < 3:
                        edges = np.linspace(lo - 1e-6, hi + 1e-6, self.n_bins + 1)

            self.bin_edges[j] = edges
            counts, _ = np.histogram(vals, bins=edges)
            alpha = 1e-3
            self.ref_bin_probs[j] = (counts + alpha) / (n_samples + alpha * len(counts))


    def _raw_covariate_score(self, ref: np.ndarray, curr: np.ndarray) -> float:
        """Compute the uncalibrated covariate score for bounded arrays."""
        if ref.ndim != 2 or curr.ndim != 2 or ref.shape[1] != curr.shape[1]:
            return 0.0
        scores: List[float] = []
        for j in range(ref.shape[1]):
            ref_vals = ref[:, j]
            curr_vals = curr[:, j]
            edges = np.percentile(ref_vals, np.linspace(0, 100, self.n_bins + 1))
            edges[0] -= 1e-6
            edges[-1] += 1e-6
            edges = np.unique(edges)
            if len(edges) < 3:
                lo, hi = float(np.min(ref_vals)), float(np.max(ref_vals))
                edges = np.linspace(lo - 1e-6, hi + 1e-6, self.n_bins + 1)
            rc, _ = np.histogram(ref_vals, bins=edges)
            cc, _ = np.histogram(curr_vals, bins=edges)
            alpha = 1e-3
            rp = (rc + alpha) / (len(ref_vals) + alpha * len(rc))
            cp = (cc + alpha) / (len(curr_vals) + alpha * len(cc))
            psi = float(max(0.0, np.sum((cp - rp) * np.log(cp / rp))))
            scale = max(1e-6, float(np.std(ref_vals)))
            wass = float(wasserstein_distance(ref_vals, curr_vals) / scale)
            scores.append(
                0.5 * np.clip(psi / max(self.psi_high, 1e-6), 0.0, 1.0)
                + 0.5 * np.clip(wass / max(self.wasserstein_threshold, 1e-6), 0.0, 1.0)
            )
        return float(np.percentile(scores, 80)) if scores else 0.0

    def fit_reference(
        self,
        X_ref: np.ndarray,
        residuals_ref: Optional[np.ndarray] = None,
    ) -> "DriftDetector":
        X_arr = np.asarray(X_ref, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X_ref must be a 2D array")
        if len(X_arr) > self.reference_size:
            idx = self.rng.choice(len(X_arr), size=self.reference_size, replace=False)
            idx.sort()
            self.reference_X = np.ascontiguousarray(X_arr[idx].copy())
            self.reference_seen = len(X_arr)
        else:
            self.reference_X = np.ascontiguousarray(X_arr.copy())
            self.reference_seen = len(X_arr)

        self._rebuild_distribution_summary()
        # Calibrate the detector against finite-sample noise in the bounded
        # reference itself.  This prevents normal sampling variation from
        # being mistaken for shift while preserving sensitivity to large shifts.
        self.null_drift_floor = 0.0
        if len(self.reference_X) >= 64:
            vals = []
            for _ in range(3):
                perm = self.rng.permutation(len(self.reference_X))
                half = len(perm) // 2
                a = self.reference_X[perm[:half]]
                b = self.reference_X[perm[half:]]
                vals.append(self._raw_covariate_score(a, b))
            self.null_drift_floor = float(np.clip(np.percentile(vals, 75), 0.0, 0.60))
        if residuals_ref is not None:
            self.cusum.initialize_baseline(residuals_ref)
        return self

    def _compute_feature_psi(self, feat_idx: int, curr_vals: np.ndarray) -> float:
        edges = self.bin_edges.get(feat_idx)
        ref_probs = self.ref_bin_probs.get(feat_idx)
        if edges is None or ref_probs is None or len(curr_vals) == 0:
            return 0.0
        curr_counts, _ = np.histogram(curr_vals, bins=edges)
        alpha = 1e-3
        curr_probs = (curr_counts + alpha) / (len(curr_vals) + alpha * len(curr_counts))
        return float(max(0.0, np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))))

    def evaluate(
        self,
        X_curr: np.ndarray,
        residuals_curr: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> DriftReport:
        X_arr = np.asarray(X_curr, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X_curr must be a 2D array")
        n_samples, n_features = X_arr.shape

        if self.reference_X is None:
            return DriftReport(0.0, "none", False, 0.0, confidence=0.0)

        if n_features != self.reference_X.shape[1]:
            raise ValueError("Feature dimension changed after reference initialization")

        names = feature_names if feature_names and len(feature_names) == n_features else [
            f"feat_{j}" for j in range(n_features)
        ]

        psi_dict: Dict[str, float] = {}
        wass_dict: Dict[str, float] = {}
        feature_scores: List[float] = []

        for j in range(n_features):
            curr = X_arr[:, j]
            ref = self.reference_X[:, j]
            psi = self._compute_feature_psi(j, curr)
            wass = float(wasserstein_distance(ref, curr) / max(1e-6, self.reference_stds[j]))
            psi_dict[names[j]] = psi
            wass_dict[names[j]] = wass

            psi_component = np.clip(psi / max(self.psi_high, 1e-6), 0.0, 1.0)
            wass_component = np.clip(wass / max(self.wasserstein_threshold, 1e-6), 0.0, 1.0)
            feature_scores.append(float(0.5 * psi_component + 0.5 * wass_component))

        raw_covariate_score = float(np.percentile(feature_scores, 80)) if feature_scores else 0.0
        if self.null_drift_floor > 0:
            covariate_score = float(np.clip(
                (raw_covariate_score - self.null_drift_floor)
                / max(1e-6, 1.0 - self.null_drift_floor),
                0.0, 1.0
            ))
        else:
            covariate_score = raw_covariate_score

        concept_detected = False
        concept_score = 0.0
        if residuals_curr is not None and n_samples >= self.min_samples_for_detection:
            concept_detected, concept_score = self.cusum.update(residuals_curr)
        concept_component = float(np.clip(concept_score, 0.0, 1.0))

        overall = float(np.clip(
            0.65 * covariate_score + 0.35 * concept_component
            if concept_component > 0
            else covariate_score,
            0.0, 1.0
        ))

        # Thresholds are configuration parameters, not domain assumptions.
        if n_samples < self.min_samples_for_detection:
            severity = "low" if overall >= self.psi_low else "none"
        elif concept_detected or overall >= 0.55:
            severity = "high"
        elif overall >= 0.25:
            severity = "medium"
        elif overall >= self.psi_low:
            severity = "low"
        else:
            severity = "none"

        if concept_detected and covariate_score >= self.psi_low:
            drift_type = "compound"
        elif concept_detected:
            drift_type = "concept"
        elif covariate_score >= self.psi_low:
            drift_type = "covariate"
        else:
            drift_type = "none"

        ranked = sorted(names, key=lambda f: max(psi_dict[f], wass_dict[f]), reverse=True)
        top = [
            f for f in ranked
            if max(psi_dict[f], wass_dict[f]) >= self.psi_low
        ][:5]

        confidence = float(1.0 - np.exp(-n_samples / self.min_samples_for_detection))

        return DriftReport(
            overall_drift_score=overall,
            severity=severity,
            is_concept_drift=concept_detected,
            concept_drift_score=concept_score,
            feature_psi=psi_dict,
            feature_wasserstein=wass_dict,
            top_drifted_features=top,
            confidence=confidence,
            covariate_drift_score=covariate_score,
            drift_type=drift_type,
        )

    def update_reference(self, X_new: np.ndarray) -> None:
        """
        Update the bounded reference reservoir.

        This is deliberately called only after the current batch has been
        evaluated and learned, so the current batch cannot influence its own
        drift report.
        """
        X_arr = np.asarray(X_new, dtype=float)
        if X_arr.ndim != 2 or len(X_arr) == 0:
            return
        if self.reference_X is None:
            self.fit_reference(X_arr)
            return
        if X_arr.shape[1] != self.reference_X.shape[1]:
            raise ValueError("Feature dimension changed in reference update")

        reservoir = self.reference_X
        for row in X_arr:
            self.reference_seen += 1
            if len(reservoir) < self.reference_size:
                reservoir = np.vstack([reservoir, row.reshape(1, -1)])
            else:
                j = int(self.rng.integers(0, self.reference_seen))
                if j < self.reference_size:
                    reservoir[j] = row
        self.reference_X = np.ascontiguousarray(reservoir, dtype=float)
        self._rebuild_distribution_summary()

    def get_memory_stats(self) -> Dict[str, int]:
        ref_bytes = 0 if self.reference_X is None else int(self.reference_X.nbytes)
        summary_bytes = sum(v.nbytes for v in self.bin_edges.values())
        summary_bytes += sum(v.nbytes for v in self.ref_bin_probs.values())
        return {
            "reference_samples": self.reference_size_current,
            "reference_capacity": self.reference_size,
            "reference_memory_bytes": ref_bytes + summary_bytes,
        }
