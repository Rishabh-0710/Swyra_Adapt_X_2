"""
Leakage-safe, bounded streaming tabular preprocessing.

The transformed feature schema is frozen after Batch 0.  This is deliberate:
continual tree learners cannot safely reinterpret historical tree split thresholds
when normalization parameters change.  Online statistics are still maintained
for diagnostics, but they are never used to construct features for the same
batch after its labels arrive.

Categoricals use fixed-width deterministic hashing instead of an unbounded
category dictionary or target encoding.  Therefore feature dimensionality and
preprocessing memory are independent of stream length.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.task import TaskSpec


class StreamingTabularPreprocessor:
    """Bounded, leakage-safe transformer for mixed tabular streams."""

    def __init__(
        self,
        task_spec: TaskSpec,
        clip_std_threshold: float = 8.0,
        categorical_hash_bins: int = 16,
    ) -> None:
        self.task_spec = task_spec
        self.clip_std_threshold = float(clip_std_threshold)
        self.categorical_hash_bins = max(4, int(categorical_hash_bins))

        # Frozen Batch-0 transform state.
        self.num_count = 0
        self.running_means: Dict[str, float] = {}
        self.running_m2: Dict[str, float] = {}
        self.impute_values: Dict[str, float] = {}
        self.transform_means: Dict[str, float] = {}
        self.transform_stds: Dict[str, float] = {}

        # Fixed-size categorical monitoring state.  No raw category keys.
        self.cat_bin_counts: Dict[str, np.ndarray] = {}
        self.cat_total_counts: Dict[str, int] = {}

        self.feature_names_out_: List[str] = []
        self.is_fitted = False

    def _convert_to_df(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return pd.DataFrame(arr, columns=[f"feat_{i}" for i in range(arr.shape[1])])

    def _convert_to_series(
        self, y: Optional[Union[pd.Series, np.ndarray]]
    ) -> Optional[pd.Series]:
        if y is None:
            return None
        if isinstance(y, pd.Series):
            return y.copy()
        return pd.Series(np.asarray(y))

    @staticmethod
    def _stable_hash(value: object) -> int:
        payload = repr(value).encode("utf-8", errors="replace")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _cat_bin(self, value: object) -> int:
        return self._stable_hash(value) % self.categorical_hash_bins

    def _update_numerical_moments(self, df: pd.DataFrame) -> None:
        """Maintain O(number_of_features) Welford summaries."""
        for col in self.task_spec.numerical_features:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue

            batch_count = int(vals.size)
            batch_mean = float(np.mean(vals))
            batch_m2 = float(np.sum((vals - batch_mean) ** 2))

            old_count = self._feature_count(col)
            if col not in self.running_means:
                self.running_means[col] = batch_mean
                self.running_m2[col] = batch_m2
                self.impute_values[col] = batch_mean
            else:
                old_mean = self.running_means[col]
                total_count = old_count + batch_count
                delta = batch_mean - old_mean
                self.running_means[col] = old_mean + delta * batch_count / total_count
                self.running_m2[col] += batch_m2 + delta * delta * old_count * batch_count / total_count
                self.impute_values[col] = self.running_means[col]

    def _feature_count(self, col: str) -> int:
        # All numerical features share the global observation count.  This is
        # sufficient for stable online moments and avoids per-column counters.
        return max(0, int(self.num_count))

    def _update_categorical_stats(self, df: pd.DataFrame) -> None:
        """Update fixed-size hashed category histograms, never target statistics."""
        for col in self.task_spec.categorical_features:
            if col not in self.cat_bin_counts:
                self.cat_bin_counts[col] = np.zeros(self.categorical_hash_bins, dtype=np.int64)
                self.cat_total_counts[col] = 0

            series = df[col] if col in df.columns else pd.Series(["__MISSING__"] * len(df))
            for value in series.astype(object).where(~series.isna(), "__MISSING__"):
                self.cat_bin_counts[col][self._cat_bin(value)] += 1
            self.cat_total_counts[col] += int(len(series))

    def _freeze_transform_state(self) -> None:
        for col in self.task_spec.numerical_features:
            mean = float(self.running_means.get(col, 0.0))
            count = max(1, self.num_count)
            var = self.running_m2.get(col, 0.0) / max(1, count - 1)
            self.transform_means[col] = mean
            self.transform_stds[col] = float(np.sqrt(max(1e-8, var)))
            self.impute_values.setdefault(col, mean)

    def _build_feature_names(self) -> None:
        names: List[str] = []
        for col in self.task_spec.numerical_features:
            names.append(f"{col}_scaled")
            names.append(f"{col}_isna")
        for col in self.task_spec.categorical_features:
            names.extend(
                f"{col}_hash_{i}" for i in range(self.categorical_hash_bins)
            )
        self.feature_names_out_ = names

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "StreamingTabularPreprocessor":
        """Fit only on Batch 0 and freeze the feature representation."""
        df = self._convert_to_df(X)

        if not self.task_spec.numerical_features and not self.task_spec.categorical_features:
            self.task_spec.numerical_features = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]
            self.task_spec.categorical_features = [
                c for c in df.columns if c not in self.task_spec.numerical_features
            ]

        self.num_count = len(df)
        self._update_numerical_moments(df)
        self._update_categorical_stats(df)
        self._freeze_transform_state()
        self._build_feature_names()
        self.is_fitted = True
        return self

    def partial_fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "StreamingTabularPreprocessor":
        """
        Observe a new batch for bounded diagnostics.

        Crucially, these updated statistics do not alter transform parameters.
        This prevents representation drift and target leakage in continual
        updates.  ``y`` is accepted for API compatibility but intentionally
        ignored.
        """
        if not self.is_fitted:
            return self.fit(X, y)

        df = self._convert_to_df(X)
        self._update_numerical_moments(df)
        self.num_count += len(df)
        self._update_categorical_stats(df)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform using only pre-update/frozen state.  No labels are consulted."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        df = self._convert_to_df(X)
        transformed: List[np.ndarray] = []

        for col in self.task_spec.numerical_features:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            else:
                series = np.full(len(df), np.nan, dtype=float)

            nan_mask = np.isnan(series)
            mean = self.transform_means.get(col, 0.0)
            std = self.transform_stds.get(col, 1.0)
            imputed = np.where(nan_mask, self.impute_values.get(col, mean), series)
            z = (imputed - mean) / max(std, 1e-8)
            z = np.clip(z, -self.clip_std_threshold, self.clip_std_threshold)

            # Always emit the missingness column so dimensionality never changes.
            transformed.append(z.reshape(-1, 1))
            transformed.append(nan_mask.astype(np.float32).reshape(-1, 1))

        for col in self.task_spec.categorical_features:
            series = df[col] if col in df.columns else pd.Series(["__MISSING__"] * len(df))
            series = series.astype(object).where(~series.isna(), "__MISSING__")
            one_hot = np.zeros((len(df), self.categorical_hash_bins), dtype=np.float32)
            for i, value in enumerate(series):
                one_hot[i, self._cat_bin(value)] = 1.0
            transformed.append(one_hot)

        if not transformed:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.hstack(transformed).astype(np.float32, copy=False)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def partial_fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        # Kept for backwards compatibility.  Importantly: transform occurs
        # before observing this batch.
        if not self.is_fitted:
            return self.fit_transform(X, y)
        X_trans = self.transform(X)
        self.partial_fit(X, y)
        return X_trans

    def get_memory_stats(self) -> Dict[str, int]:
        cat_bytes = sum(v.nbytes for v in self.cat_bin_counts.values())
        return {
            "categorical_summary_bytes": int(cat_bytes),
            "n_categorical_features": len(self.task_spec.categorical_features),
            "hash_bins_per_categorical": self.categorical_hash_bins,
            "feature_count": len(self.feature_names_out_),
        }
