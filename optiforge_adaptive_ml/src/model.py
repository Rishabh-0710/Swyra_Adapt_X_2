"""
ADAPT-X continual learner.

Architecture:
    Initial Learning
        -> Persistent Model State
        -> Pre-update Prediction
        -> DistributionShiftDetector
        -> AdaptationController
        -> Controlled Incremental Update
        -> BoundedReplayMemory
        -> Next Batch

The stable anchor is a persistent LightGBM booster extended with new trees
using ``init_model``.  The adaptive specialist is another persistent booster
that can be compactly reset when its bounded tree budget is exhausted or a
severe concept shift makes a fresh recent expert useful.  Neither component
re-fits the historical stream from scratch.
"""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Deque, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.adaptation import AdaptationAction, AdaptationController
from src.drift import DriftDetector, DriftReport
from src.memory import StratifiedDensityReplayBuffer
from src.preprocessing import StreamingTabularPreprocessor
from src.task import TaskSpec, infer_task_spec
from src.utils import get_deep_memory_bytes


class OptiForgeAdaptiveModel:
    """
    Domain-agnostic continual learner for supervised tabular streams.

    Historical raw batches are never retained.  Only bounded replay samples,
    bounded statistical references, frozen preprocessing parameters, persistent
    model state, and bounded diagnostics survive an update.
    """

    def __init__(
        self,
        task_spec: Optional[TaskSpec] = None,
        max_replay_samples: int = 500,
        base_learning_rate: float = 0.05,
        base_trees: int = 25,
        max_trees_anchor: int = 180,
        max_trees_specialist: int = 80,
        initial_anchor_trees: int = 100,
        initial_specialist_trees: int = 25,
        l2_regularization: float = 1.5,
        random_seed: int = 42,
        reference_size: int = 256,
        diagnostic_history_size: int = 32,
    ) -> None:
        self.task_spec = task_spec
        self.max_replay_samples = max(1, int(max_replay_samples))
        self.base_learning_rate = float(base_learning_rate)
        self.base_trees = max(5, int(base_trees))
        self.max_trees_anchor = max(self.base_trees, int(max_trees_anchor))
        self.max_trees_specialist = max(self.base_trees, int(max_trees_specialist))
        self.initial_anchor_trees = min(int(initial_anchor_trees), self.max_trees_anchor)
        self.initial_specialist_trees = min(int(initial_specialist_trees), self.max_trees_specialist)
        self.l2_regularization = float(l2_regularization)
        self.random_seed = int(random_seed)
        self.reference_size = max(32, int(reference_size))
        self.rng = np.random.default_rng(self.random_seed)

        self.preprocessor: Optional[StreamingTabularPreprocessor] = None
        self.replay_buffer: Optional[StratifiedDensityReplayBuffer] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.adaptation_controller: Optional[AdaptationController] = None

        self.anchor_model: Optional[Any] = None
        self.specialist_model: Optional[Any] = None

        self.anchor_weight = 1.0
        self.specialist_weight = 0.0

        self.is_initialized = False
        self.update_count = 0
        self.total_samples_seen = 0

        # Diagnostic histories are bounded.  Lifetime counters remain scalars.
        self.drift_history: Deque[DriftReport] = deque(maxlen=diagnostic_history_size)
        self.adaptation_history: Deque[AdaptationAction] = deque(maxlen=diagnostic_history_size)
        self.update_times: Deque[float] = deque(maxlen=diagnostic_history_size)

    def _build_base_estimator(
        self,
        n_estimators: int,
        learning_rate: float,
        l2_reg: float,
    ) -> Any:
        assert self.task_spec is not None
        common = dict(
            learning_rate=float(learning_rate),
            reg_lambda=float(l2_reg),
            random_state=self.random_seed,
            verbose=-1,
            n_jobs=1,
        )
        try:
            if self.task_spec.task_type == "regression":
                return LGBMRegressor(n_estimators=max(1, int(n_estimators)), **common)
            return LGBMClassifier(n_estimators=max(1, int(n_estimators)), **common)
        except Exception:
            if self.task_spec.task_type == "regression":
                return HistGradientBoostingRegressor(
                    max_iter=max(1, int(n_estimators)),
                    learning_rate=float(learning_rate),
                    l2_regularization=float(l2_reg),
                    random_state=self.random_seed,
                )
            return HistGradientBoostingClassifier(
                max_iter=max(1, int(n_estimators)),
                learning_rate=float(learning_rate),
                l2_regularization=float(l2_reg),
                random_state=self.random_seed,
            )

    @staticmethod
    def _tree_count(model: Optional[Any]) -> int:
        if model is None:
            return 0
        booster = getattr(model, "booster_", None)
        if booster is not None:
            try:
                return int(booster.num_trees())
            except Exception:
                pass
        return int(getattr(model, "n_estimators", 0) or getattr(model, "max_iter", 0))

    def _fit_incremental(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        additional_trees: int,
        learning_rate: float,
        l2_reg: float,
    ) -> Any:
        """
        Append trees to a persistent LightGBM booster.

        If the LightGBM backend is unavailable, fall back to fitting a bounded
        model on replay + current evidence.  The normal path is genuinely
        incremental because ``init_model`` preserves all existing trees.
        """
        additional_trees = max(1, int(additional_trees))
        if hasattr(model, "booster_"):
            updated = self._build_base_estimator(additional_trees, learning_rate, l2_reg)
            try:
                updated.fit(X, y, init_model=model.booster_)
                return updated
            except Exception:
                pass

        updated = self._build_base_estimator(additional_trees, learning_rate, l2_reg)
        updated.fit(X, y)
        return updated

    def initialize(
        self,
        X_initial: Union[pd.DataFrame, np.ndarray],
        y_initial: Union[pd.Series, np.ndarray],
        task_spec: Optional[TaskSpec] = None,
    ) -> "OptiForgeAdaptiveModel":
        t0 = time.perf_counter()

        if task_spec is not None:
            self.task_spec = task_spec
        elif self.task_spec is None:
            self.task_spec = infer_task_spec(X_initial, y_initial)

        self.preprocessor = StreamingTabularPreprocessor(self.task_spec)
        self.replay_buffer = StratifiedDensityReplayBuffer(
            task_spec=self.task_spec,
            max_samples=self.max_replay_samples,
            random_seed=self.random_seed,
        )
        self.drift_detector = DriftDetector(
            reference_size=self.reference_size,
            random_seed=self.random_seed,
        )
        self.adaptation_controller = AdaptationController(
            base_lr=self.base_learning_rate,
            base_trees=self.base_trees,
            base_l2=self.l2_regularization,
        )

        # Batch 0 is the only initial fitting operation.
        X_trans = self.preprocessor.fit_transform(X_initial)
        y_arr = np.asarray(y_initial, dtype=float)

        self.anchor_model = self._build_base_estimator(
            self.initial_anchor_trees,
            self.base_learning_rate,
            self.l2_regularization * 1.5,
        )
        self.anchor_model.fit(X_trans, y_arr)

        self.specialist_model = self._build_base_estimator(
            self.initial_specialist_trees,
            self.base_learning_rate,
            self.l2_regularization,
        )
        self.specialist_model.fit(X_trans, y_arr)

        if self.task_spec.task_type == "regression":
            residuals = y_arr - self.anchor_model.predict(X_trans)
        else:
            residuals = y_arr - self.anchor_model.predict_proba(X_trans)[:, 1]

        self.drift_detector.fit_reference(X_trans, residuals)
        self.replay_buffer.update(X_trans, y_arr)

        self.anchor_weight = 1.0
        self.specialist_weight = 0.0
        self.is_initialized = True
        self.update_count = 0
        self.total_samples_seen = len(y_arr)
        self.update_times.clear()
        self.update_times.append(time.perf_counter() - t0)
        return self

    def _sample_current(self, X: np.ndarray, y: np.ndarray, max_n: int) -> tuple[np.ndarray, np.ndarray]:
        if len(X) <= max_n:
            return X, y
        idx = self.rng.choice(len(X), size=max_n, replace=False)
        return X[idx], y[idx]

    def update(
        self,
        X_batch: Union[pd.DataFrame, np.ndarray],
        y_batch: Union[pd.Series, np.ndarray],
    ) -> "OptiForgeAdaptiveModel":
        if (
            not self.is_initialized
            or self.preprocessor is None
            or self.replay_buffer is None
            or self.drift_detector is None
            or self.adaptation_controller is None
        ):
            raise RuntimeError("Model must be initialized before update().")

        t0 = time.perf_counter()
        self.update_count += 1
        y_arr = np.asarray(y_batch, dtype=float)
        self.total_samples_seen += len(y_arr)

        # ------------------------------------------------------------
        # 1. TRANSFORM -> PREDICT -> DETECT, before any state update.
        # ------------------------------------------------------------
        X_trans = self.preprocessor.transform(X_batch)
        if self.task_spec.task_type == "regression":
            residuals = y_arr - self._predict_transformed(X_trans)
        else:
            residuals = y_arr - self._predict_proba_transformed(X_trans)[:, 1]

        report = self.drift_detector.evaluate(
            X_trans,
            residuals_curr=residuals,
            feature_names=self.preprocessor.feature_names_out_,
        )
        self.drift_history.append(report)

        # ------------------------------------------------------------
        # 2. ADAPTATION CONTROLLER decides plasticity/stability.
        # ------------------------------------------------------------
        action = self.adaptation_controller.compute_action(report)
        self.adaptation_history.append(action)

        # Replay uses the same frozen representation as the current batch.
        X_rep, y_rep = self.replay_buffer.get_data()
        has_replay = len(X_rep) > 0

        # Specialist receives more current evidence.  Replay remains bounded.
        n_rep_spec = min(
            len(X_rep),
            max(8, int(len(X_trans) * action.replay_blend_ratio)),
        ) if has_replay else 0
        if n_rep_spec:
            rep_idx = self.rng.choice(len(X_rep), size=n_rep_spec, replace=False)
            X_spec = np.vstack((X_trans, X_rep[rep_idx]))
            y_spec = np.concatenate((y_arr, y_rep[rep_idx]))
        else:
            X_spec, y_spec = X_trans, y_arr

        # Stable anchor receives bounded replay plus a conservative slice of
        # new evidence. It never sees the full historical stream.
        if has_replay:
            n_current_anchor = max(
                8, min(len(X_trans), int(len(X_trans) * max(0.10, 0.20 * action.adaptation_rate)))
            )
            X_cur_anchor, y_cur_anchor = self._sample_current(
                X_trans, y_arr, n_current_anchor
            )
            X_anchor = np.vstack((X_rep, X_cur_anchor))
            y_anchor = np.concatenate((y_rep, y_cur_anchor))
        else:
            X_anchor, y_anchor = self._sample_current(
                X_trans, y_arr, max(8, int(len(X_trans) * 0.5))
            )

        # ------------------------------------------------------------
        # 3. PERSISTENT INCREMENTAL MODEL UPDATE.
        # ------------------------------------------------------------
        current_anchor_trees = self._tree_count(self.anchor_model)
        anchor_budget = self.max_trees_anchor - current_anchor_trees
        if anchor_budget > 0:
            add_anchor = min(anchor_budget, max(1, action.n_new_trees // 2))
            self.anchor_model = self._fit_incremental(
                self.anchor_model,
                X_anchor,
                y_anchor,
                add_anchor,
                self.base_learning_rate * 0.8,
                self.l2_regularization * 1.8,
            )

        current_spec_trees = self._tree_count(self.specialist_model)
        # On a severe concept shift, a bounded specialist reset is intentional:
        # it is the plastic component, while the anchor remains persistent.
        reset_specialist = bool(report.is_concept_drift and report.severity == "high")
        if reset_specialist:
            self.specialist_model = self._build_base_estimator(
                min(self.base_trees * 2, self.max_trees_specialist),
                action.learning_rate,
                action.l2_regularization,
            )
            self.specialist_model.fit(X_spec, y_spec)
        else:
            spec_budget = self.max_trees_specialist - current_spec_trees
            if spec_budget > 0:
                add_spec = min(spec_budget, max(1, action.n_new_trees))
                self.specialist_model = self._fit_incremental(
                    self.specialist_model,
                    X_spec,
                    y_spec,
                    add_spec,
                    action.learning_rate,
                    action.l2_regularization,
                )
            elif report.severity in ("medium", "high"):
                # Compact recent-head refresh when the specialist reaches its
                # bounded capacity. This is not full historical retraining.
                self.specialist_model = self._build_base_estimator(
                    min(self.base_trees * 2, self.max_trees_specialist),
                    action.learning_rate,
                    action.l2_regularization,
                )
                self.specialist_model.fit(X_spec, y_spec)

        # ------------------------------------------------------------
        # 4. AFTER learning, update bounded preprocessing/reference state.
        # ------------------------------------------------------------
        self.replay_buffer.update(X_trans, y_arr)
        self.preprocessor.partial_fit(X_batch)  # diagnostics only; transform stays frozen
        self.drift_detector.update_reference(X_trans)

        self.anchor_weight = float(action.anchor_weight)
        self.specialist_weight = float(action.specialist_weight)

        self.update_times.append(time.perf_counter() - t0)
        return self

    def _predict_transformed(self, X_trans: np.ndarray) -> np.ndarray:
        p_anchor = self.anchor_model.predict(X_trans)
        if self.specialist_model is not None and self.specialist_weight > 0.0:
            p_spec = self.specialist_model.predict(X_trans)
            return self.anchor_weight * p_anchor + self.specialist_weight * p_spec
        return p_anchor

    def _predict_proba_transformed(self, X_trans: np.ndarray) -> np.ndarray:
        p_anchor = self.anchor_model.predict_proba(X_trans)
        if self.specialist_model is not None and self.specialist_weight > 0.0:
            p_spec = self.specialist_model.predict_proba(X_trans)
            blended = self.anchor_weight * p_anchor + self.specialist_weight * p_spec
            return blended / np.maximum(np.sum(blended, axis=1, keepdims=True), 1e-12)
        return p_anchor

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_initialized or self.preprocessor is None:
            raise RuntimeError("Model is not initialized.")
        X_trans = self.preprocessor.transform(X)
        if self.task_spec.task_type == "regression":
            return self._predict_transformed(X_trans)
        return (self._predict_proba_transformed(X_trans)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_initialized or self.preprocessor is None:
            raise RuntimeError("Model is not initialized.")
        if self.task_spec.task_type != "binary_classification":
            raise ValueError("predict_proba is only available for classification tasks.")
        return self._predict_proba_transformed(self.preprocessor.transform(X))

    def get_memory_stats(self) -> Dict[str, Any]:
        replay_stats = self.replay_buffer.get_memory_stats() if self.replay_buffer else {}
        drift_stats = self.drift_detector.get_memory_stats() if self.drift_detector else {}
        prep_stats = self.preprocessor.get_memory_stats() if self.preprocessor else {}

        # All retained structures are bounded.  This is an accounting estimate,
        # not RSS of the Python process.
        component_bytes = (
            int(replay_stats.get("memory_bytes", 0))
            + int(drift_stats.get("reference_memory_bytes", 0))
            + int(get_deep_memory_bytes(self.preprocessor)) if self.preprocessor else 0
        )
        if self.anchor_model is not None:
            component_bytes += int(get_deep_memory_bytes(self.anchor_model))
        if self.specialist_model is not None:
            component_bytes += int(get_deep_memory_bytes(self.specialist_model))

        return {
            "memory_bytes": int(component_bytes),
            "replay_buffer": replay_stats,
            "drift_reference": drift_stats,
            "preprocessor": prep_stats,
            "total_samples_observed": self.total_samples_seen,
            "update_count": self.update_count,
            "anchor_trees": self._tree_count(self.anchor_model),
            "specialist_trees": self._tree_count(self.specialist_model),
            "diagnostic_history_capacity": self.drift_history.maxlen,
            "gating_weights": {
                "anchor": round(self.anchor_weight, 3),
                "specialist": round(self.specialist_weight, 3),
            },
        }

    def get_drift_history(self) -> List[DriftReport]:
        return list(self.drift_history)

    def get_adaptation_history(self) -> List[AdaptationAction]:
        return list(self.adaptation_history)
