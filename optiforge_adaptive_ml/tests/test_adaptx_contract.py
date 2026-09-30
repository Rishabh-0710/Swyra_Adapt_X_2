
"""ADAPT-X contract tests: bounded state, leakage safety, and sequential semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.drift import DriftDetector
from src.model import OptiForgeAdaptiveModel
from src.preprocessing import StreamingTabularPreprocessor
from src.task import TaskSpec


def test_reference_memory_is_hard_bounded():
    rng = np.random.default_rng(7)
    detector = DriftDetector(reference_size=64, random_seed=7)
    detector.fit_reference(rng.normal(size=(5000, 12)))
    assert detector.reference_size_current <= 64

    for _ in range(20):
        detector.update_reference(rng.normal(size=(1000, 12)))
        assert detector.reference_size_current <= 64


def test_target_labels_cannot_change_current_batch_features():
    """A preprocessing transform must be independent of the current y batch."""
    spec = TaskSpec(task_type="regression", numerical_features=["x"], categorical_features=["cat"])
    X0 = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "cat": ["a", "b", "a", "b"]})
    p1 = StreamingTabularPreprocessor(spec)
    p2 = StreamingTabularPreprocessor(spec)

    p1.fit(X0, np.array([0.0, 0.0, 1.0, 1.0]))
    p2.fit(X0, np.array([100.0, 100.0, -100.0, -100.0]))

    X_batch = pd.DataFrame({"x": [4.0, 5.0], "cat": ["a", "new"]})
    np.testing.assert_array_equal(
        p1.transform(X_batch),
        p2.transform(X_batch),
        err_msg="Current labels altered the feature representation",
    )


def test_feature_schema_stays_stable_across_missingness_changes():
    spec = TaskSpec(task_type="regression", numerical_features=["x"], categorical_features=["cat"])
    p = StreamingTabularPreprocessor(spec)
    p.fit(
        pd.DataFrame({"x": [1.0, 2.0], "cat": ["a", "b"]}),
        np.array([1.0, 2.0]),
    )
    width = len(p.feature_names_out_)

    a = p.transform(pd.DataFrame({"x": [3.0], "cat": ["c"]}))
    b = p.transform(pd.DataFrame({"x": [np.nan], "cat": [None]}))
    assert a.shape[1] == b.shape[1] == width


def test_diagnostics_are_bounded_on_long_stream():
    rng = np.random.default_rng(11)
    model = OptiForgeAdaptiveModel(
        max_replay_samples=32,
        reference_size=32,
        diagnostic_history_size=8,
        random_seed=11,
    )
    X0 = pd.DataFrame(rng.normal(size=(80, 4)), columns=["a", "b", "c", "d"])
    y0 = pd.Series(X0["a"] * 2.0)
    model.initialize(X0, y0)

    for _ in range(40):
        X = pd.DataFrame(rng.normal(size=(80, 4)), columns=["a", "b", "c", "d"])
        y = pd.Series(X["a"] * 2.0)
        model.update(X, y)

    assert len(model.drift_history) <= 8
    assert len(model.adaptation_history) <= 8
    assert len(model.update_times) <= 8
    assert model.replay_buffer.current_size <= 32
    assert model.drift_detector.reference_size_current <= 32


def test_model_state_is_extended_incrementally():
    rng = np.random.default_rng(21)
    X0 = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y0 = pd.Series(2 * X0["a"] - X0["b"])
    model = OptiForgeAdaptiveModel(random_seed=21, max_replay_samples=64)
    model.initialize(X0, y0)
    initial_anchor = model._tree_count(model.anchor_model)
    initial_specialist = model._tree_count(model.specialist_model)

    X1 = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y1 = pd.Series(2 * X1["a"] - X1["b"])
    model.update(X1, y1)

    assert model._tree_count(model.anchor_model) >= initial_anchor
    assert model._tree_count(model.specialist_model) >= initial_specialist
    assert model.update_count == 1
