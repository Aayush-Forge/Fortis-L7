"""
Synthetic request generator: samples the 5D fingerprint by traffic tier.
Agent never sees tier or ground-truth label directly — only env internals.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

TrafficTier = Literal["human", "easy", "medium", "hard"]


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


class RequestGenerator:
    """
    Draws features in architect-defined bands for humans and three bot levels.
    `difficulty_level` biases bot subtype mix when sampling bots (1=easy-heavy ... 3=hard-heavy).
    """

    def __init__(
        self,
        level: int = 1,
        human_prior: float = 0.35,
        seed: Optional[int] = None,
    ):
        self.level = int(np.clip(level, 1, 3))
        self.human_prior = float(np.clip(human_prior, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

    def _sample_human(self) -> Dict[str, float]:
        """Low threat IP, human-like path and jitter; RS should stay in verified/probably human."""
        rng = self.rng
        return {
            "ip_reputation": _uniform(rng, 0.0, 0.12),
            "velocity_score": _uniform(rng, 0.05, 0.32),
            "entropy_level": _uniform(rng, 0.15, 0.45),
            "navigation_path_index": _uniform(rng, 0.70, 0.98),
            "jitter_value": _uniform(rng, 0.45, 0.90),
        }

    def _sample_easy_bot(self) -> Dict[str, float]:
        rng = self.rng
        return {
            "ip_reputation": _uniform(rng, 0.75, 0.95),
            "velocity_score": _uniform(rng, 0.85, 1.0),
            "entropy_level": _uniform(rng, 0.10, 0.25),
            "navigation_path_index": _uniform(rng, 0.10, 0.20),
            "jitter_value": _uniform(rng, 0.02, 0.08),
        }

    def _sample_medium_bot(self) -> Dict[str, float]:
        rng = self.rng
        return {
            "ip_reputation": _uniform(rng, 0.30, 0.65),
            "velocity_score": _uniform(rng, 0.50, 0.75),
            "entropy_level": _uniform(rng, 0.35, 0.55),
            "navigation_path_index": _uniform(rng, 0.30, 0.55),
            "jitter_value": _uniform(rng, 0.20, 0.40),
        }

    def _sample_hard_bot(self) -> Dict[str, float]:
        rng = self.rng
        return {
            "ip_reputation": _uniform(rng, 0.10, 0.40),
            "velocity_score": _uniform(rng, 0.20, 0.50),
            "entropy_level": _uniform(rng, 0.55, 0.75),
            "navigation_path_index": _uniform(rng, 0.60, 0.80),
            "jitter_value": _uniform(rng, 0.35, 0.60),
        }

    def _pick_bot_tier(self) -> TrafficTier:
        """Higher env level increases probability of harder bot profiles."""
        rng = self.rng
        if self.level <= 1:
            p = np.array([0.70, 0.22, 0.08], dtype=np.float64)
        elif self.level == 2:
            p = np.array([0.25, 0.50, 0.25], dtype=np.float64)
        else:
            p = np.array([0.10, 0.30, 0.60], dtype=np.float64)
        choice = rng.choice(["easy", "medium", "hard"], p=p)
        return choice  # type: ignore[return-value]

    def sample_fingerprint(self) -> Tuple[Dict[str, float], TrafficTier]:
        """
        Returns (feature_dict, tier). Tier is for env logging only.
        """
        if self.rng.random() < self.human_prior:
            return self._sample_human(), "human"
        tier = self._pick_bot_tier()
        if tier == "easy":
            return self._sample_easy_bot(), "easy"
        if tier == "medium":
            return self._sample_medium_bot(), "medium"
        return self._sample_hard_bot(), "hard"

    # Back-compat name used by older env
    def generate_request(self, current_cpu: float | None = None) -> Dict[str, Any]:
        feats, tier = self.sample_fingerprint()
        out = dict(feats)
        if current_cpu is not None:
            out["_server_cpu_load"] = float(current_cpu)
        out["_tier"] = tier
        return out
