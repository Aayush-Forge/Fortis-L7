"""
Hidden Risk Score (RS) engine — architect weights. The agent never observes RS directly.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import numpy as np
from pydantic import BaseModel, Field

ClassificationLabel = Literal[
    "hard_bot",
    "soft_bot",
    "ambiguous",
    "probably_human",
    "verified_human",
]


class ObservationVector(BaseModel):
    """Five-dimensional digital fingerprint of a single API request (agent-visible)."""

    ip_reputation: float = Field(ge=0.0, le=1.0, description="Threat score; 1.0 = blacklisted")
    velocity_score: float = Field(ge=0.0, le=1.0)
    entropy_level: float = Field(ge=0.0, le=1.0)
    navigation_path_index: float = Field(ge=0.0, le=1.0, description="Higher = more human-like path")
    jitter_value: float = Field(ge=0.0, le=1.0, description="Higher = more human-like timing")

    def as_numpy(self) -> np.ndarray:
        return np.array(
            [
                self.ip_reputation,
                self.velocity_score,
                self.entropy_level,
                self.navigation_path_index,
                self.jitter_value,
            ],
            dtype=np.float32,
        )


class RiskEngineConfig(BaseModel):
    """Default weights — do not change without architect sign-off."""

    w_ip: float = 0.30
    w_vel: float = 0.25
    w_ent: float = 0.20
    w_nav: float = 0.15
    w_jit: float = 0.10


class RiskEngine:
    """
    Computes RS and discrete labels for reward / logging.
    RL agent must infer risk from ObservationVector only.
    """

    THRESHOLDS: Dict[str, float] = {
        "hard_bot": 0.80,
        "soft_bot": 0.60,
        "ambiguous": 0.40,
        "probably_human": 0.20,
    }

    # Actions: 0 Allow, 1 Throttle, 2 Challenge, 3 Block
    EXPECTED_ACTION: Dict[ClassificationLabel, int] = {
        "hard_bot": 3,
        "soft_bot": 2,
        "ambiguous": 1,
        "probably_human": 0,
        "verified_human": 0,
    }

    # Calibrated weights: R = w1*TP - w2*FN - w3*FP (used when matrix flags apply)
    w1: float = 1.0
    w2: float = 2.0
    w3: float = 10.0

    def __init__(self, config: RiskEngineConfig | None = None):
        self.config = config or RiskEngineConfig()
        cfg = self.config
        s = cfg.w_ip + cfg.w_vel + cfg.w_ent + cfg.w_nav + cfg.w_jit
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"RiskEngineConfig weights must sum to 1.0, got {s}")

    def compute_rs(self, obs: ObservationVector) -> float:
        nav_inv = 1.0 - obs.navigation_path_index
        c = self.config
        rs = (
            c.w_ip * obs.ip_reputation
            + c.w_vel * obs.velocity_score
            + c.w_ent * obs.entropy_level
            + c.w_nav * nav_inv
            + c.w_jit * obs.jitter_value
        )
        return float(np.clip(rs, 0.0, 1.0))

    def classify(self, rs: float) -> ClassificationLabel:
        t = self.THRESHOLDS
        if rs >= t["hard_bot"]:
            return "hard_bot"
        if rs >= t["soft_bot"]:
            return "soft_bot"
        if rs >= t["ambiguous"]:
            return "ambiguous"
        if rs >= t["probably_human"]:
            return "probably_human"
        return "verified_human"

    def compute(self, obs: ObservationVector) -> Tuple[float, ClassificationLabel]:
        rs = self.compute_rs(obs)
        return rs, self.classify(rs)

    def expected_action(self, label: ClassificationLabel) -> int:
        return self.EXPECTED_ACTION[label]

    def compute_reward(
        self, action: int, label: ClassificationLabel
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Dense reward table (5 labels x 4 actions) aligned with architect intent:
        strong bot catch (block/challenge), nuanced handling for ambiguous,
        heavy penalty for disrupting humans.
        Returns (reward, diagnostic flags).
        """
        # Table: (label, action) -> reward
        table: Dict[Tuple[str, int], float] = {
            ("verified_human", 0): 0.5,
            ("verified_human", 1): -0.2,
            ("verified_human", 2): -2.0,
            ("verified_human", 3): -10.0,
            ("probably_human", 0): 0.4,
            ("probably_human", 1): -0.15,
            ("probably_human", 2): -1.5,
            ("probably_human", 3): -10.0,
            ("ambiguous", 0): -0.1,
            ("ambiguous", 1): 0.8,
            ("ambiguous", 2): 0.5,
            ("ambiguous", 3): 0.2,
            ("soft_bot", 0): -2.0,
            ("soft_bot", 1): 0.5,
            ("soft_bot", 2): 1.0,
            ("soft_bot", 3): 1.0,
            ("hard_bot", 0): -2.0,
            ("hard_bot", 1): 0.3,
            ("hard_bot", 2): 1.0,
            ("hard_bot", 3): 1.0,
        }
        reward = table[(label, action)]

        human = label in ("verified_human", "probably_human")
        malicious = label in ("hard_bot", "soft_bot")

        tp = malicious and action in (2, 3)
        fn = malicious and action == 0
        fp = human and action in (2, 3)
        tn = human and action == 0

        formula_r = self.w1 * float(tp) - self.w2 * float(fn) - self.w3 * float(fp)
        # Blend: table gives dense shaping; formula available for logging
        flags: Dict[str, Any] = {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
            "formula_reward": formula_r,
        }
        return reward, flags


def observation_from_array(arr: np.ndarray) -> ObservationVector:
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.shape != (5,):
        raise ValueError(f"Expected shape (5,), got {a.shape}")
    return ObservationVector(
        ip_reputation=float(a[0]),
        velocity_score=float(a[1]),
        entropy_level=float(a[2]),
        navigation_path_index=float(a[3]),
        jitter_value=float(a[4]),
    )
