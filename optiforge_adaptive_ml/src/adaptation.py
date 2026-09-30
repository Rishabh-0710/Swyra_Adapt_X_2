"""
Adaptation controller for the ADAPT-X stability/plasticity trade-off.

The controller maps measured drift intensity and drift type to a continuous
adaptation policy.  It does not inspect feature names or domain semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.drift import DriftReport


@dataclass
class AdaptationAction:
    adaptation_rate: float
    replay_blend_ratio: float
    specialist_weight: float
    anchor_weight: float
    n_new_trees: int
    learning_rate: float
    l2_regularization: float
    strategy_description: str


class AdaptationController:
    """Translate drift evidence into a bounded stability/plasticity policy."""

    def __init__(
        self,
        base_lr: float = 0.05,
        base_trees: int = 25,
        base_l2: float = 1.5,
        min_adaptation_rate: float = 0.10,
        max_adaptation_rate: float = 1.00,
        min_replay_ratio: float = 0.15,
        max_replay_ratio: float = 0.50,
    ) -> None:
        self.base_lr = float(base_lr)
        self.base_trees = max(5, int(base_trees))
        self.base_l2 = float(base_l2)
        self.min_adaptation_rate = float(min_adaptation_rate)
        self.max_adaptation_rate = float(max_adaptation_rate)
        self.min_replay_ratio = float(min_replay_ratio)
        self.max_replay_ratio = float(max_replay_ratio)

    def compute_action(self, drift_report: DriftReport) -> AdaptationAction:
        score = float(np.clip(drift_report.overall_drift_score, 0.0, 1.0))
        concept = float(np.clip(drift_report.concept_drift_score, 0.0, 1.0))
        confidence = float(np.clip(drift_report.confidence, 0.0, 1.0))

        # Confidence tempers aggressive reactions to tiny batches.
        effective = float(np.clip(score * (0.5 + 0.5 * confidence), 0.0, 1.0))
        if drift_report.is_concept_drift:
            effective = float(np.clip(effective + 0.20 * max(concept, 0.5), 0.0, 1.0))

        adaptation_rate = self.min_adaptation_rate + (
            self.max_adaptation_rate - self.min_adaptation_rate
        ) * effective

        specialist_weight = float(np.clip(
            0.08 + 0.78 * effective,
            0.05, 0.85
        ))
        # Concept drift is evidence that the mapping itself changed, so it
        # deserves extra plasticity.  Anchor weight never becomes negative.
        if drift_report.is_concept_drift:
            specialist_weight = min(0.90, specialist_weight + 0.10)
        anchor_weight = 1.0 - specialist_weight

        replay_ratio = self.max_replay_ratio - (
            self.max_replay_ratio - self.min_replay_ratio
        ) * effective
        if drift_report.is_concept_drift:
            replay_ratio = max(self.min_replay_ratio, replay_ratio - 0.05)

        n_new_trees = max(5, int(round(self.base_trees * (0.60 + adaptation_rate))))
        learning_rate = self.base_lr * (0.75 + 0.75 * adaptation_rate)
        l2 = self.base_l2 * (1.35 - 0.55 * adaptation_rate)

        if drift_report.drift_type == "none":
            desc = "Stable stream: conservative incremental update with strong replay."
        elif drift_report.drift_type == "covariate":
            desc = "Covariate shift: increase plasticity while retaining representative history."
        elif drift_report.drift_type == "concept":
            desc = "Concept shift: prioritize the adaptive component and reduce stale replay influence."
        else:
            desc = "Compound shift: strongest controlled adaptation with bounded historical retention."

        return AdaptationAction(
            adaptation_rate=float(adaptation_rate),
            replay_blend_ratio=float(replay_ratio),
            specialist_weight=float(specialist_weight),
            anchor_weight=float(anchor_weight),
            n_new_trees=n_new_trees,
            learning_rate=float(learning_rate),
            l2_regularization=float(l2),
            strategy_description=desc,
        )
