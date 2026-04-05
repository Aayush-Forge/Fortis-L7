"""
Gymnasium-compatible Fortis-L7 / DoWGuard-style environment.
Observation: 5D fingerprint only. RS and labels are internal.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data_generator import RequestGenerator, TrafficTier
from risk_engine import ObservationVector, RiskEngine


class FortisL7Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        difficulty_level: int = 1,
        max_episode_steps: int = 200,
        human_prior: float = 0.35,
        seed: Optional[int] = None,
        use_cpu_dynamics: bool = True,
    ):
        super().__init__()
        self.difficulty_level = int(difficulty_level)
        self.max_episode_steps = int(max_episode_steps)
        self.use_cpu_dynamics = use_cpu_dynamics

        self.generator = RequestGenerator(
            level=self.difficulty_level,
            human_prior=human_prior,
            seed=seed,
        )
        self.risk_engine = RiskEngine()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self._current_obs: Optional[ObservationVector] = None
        self._current_tier: TrafficTier = "human"
        self._cpu_load = 0.10
        self._step_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.generator.rng = np.random.default_rng(seed + 17)

        self._step_count = 0
        self._cpu_load = 0.10
        feats, tier = self.generator.sample_fingerprint()
        self._current_obs = ObservationVector(**feats)
        self._current_tier = tier

        rs, label = self.risk_engine.compute(self._current_obs)
        info = self._build_info(rs, label, phase="reset")
        return self._current_obs.as_numpy(), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._current_obs is None:
            raise RuntimeError("Call reset() before step().")

        assert self.action_space.contains(action)
        rs, label = self.risk_engine.compute(self._current_obs)
        acted_tier = self._current_tier
        reward, flags = self.risk_engine.compute_reward(int(action), label)

        if self.use_cpu_dynamics:
            if action == 0 and label in ("hard_bot", "soft_bot"):
                self._cpu_load = min(1.0, self._cpu_load + 0.12)
            elif action == 0 and label in ("verified_human", "probably_human"):
                self._cpu_load = max(0.05, self._cpu_load - 0.04)
            elif action == 0 and label == "ambiguous":
                self._cpu_load = min(1.0, self._cpu_load + 0.04)

        self._step_count += 1
        terminated = bool(self._cpu_load >= 1.0)
        truncated = self._step_count >= self.max_episode_steps

        feats, tier = self.generator.sample_fingerprint()
        self._current_obs = ObservationVector(**feats)
        self._current_tier = tier

        next_rs, next_label = self.risk_engine.compute(self._current_obs)
        info = self._build_info(
            rs,
            label,
            phase="step",
            action=action,
            reward=reward,
            flags=flags,
            acted_traffic_tier=acted_tier,
        )
        info["next_risk_score"] = next_rs
        info["next_classification"] = next_label
        info["next_traffic_tier"] = tier
        info["cpu_load"] = self._cpu_load

        return self._current_obs.as_numpy(), float(reward), terminated, truncated, info

    def _build_info(
        self,
        rs: float,
        label: str,
        phase: str,
        action: Optional[int] = None,
        reward: Optional[float] = None,
        flags: Optional[Dict[str, Any]] = None,
        acted_traffic_tier: Optional[TrafficTier] = None,
    ) -> Dict[str, Any]:
        expected = self.risk_engine.expected_action(label)  # type: ignore[arg-type]
        tier = acted_traffic_tier if acted_traffic_tier is not None else self._current_tier
        out: Dict[str, Any] = {
            "risk_score": rs,
            "classification": label,
            "expected_action": expected,
            "traffic_tier": tier,
            "cpu_load": self._cpu_load,
            "phase": phase,
        }
        if action is not None:
            out["action"] = action
        if reward is not None:
            out["reward"] = reward
        if flags is not None:
            out.update(flags)
        return out

    def state(self) -> Dict[str, Any]:
        """OpenEnv-style full snapshot for judges / debugging."""
        snap: Dict[str, Any] = {
            "step": self._step_count,
            "cpu_load": self._cpu_load,
            "difficulty_level": self.difficulty_level,
            "current_traffic_tier": self._current_tier,
        }
        if self._current_obs is not None:
            rs, label = self.risk_engine.compute(self._current_obs)
            snap["current_observation_dict"] = self._current_obs.model_dump()
            snap["risk_score"] = rs
            snap["classification"] = label
            snap["expected_action"] = self.risk_engine.expected_action(label)  # type: ignore[arg-type]
        return snap


# Alias for spec documents
DoWGuardEnv = FortisL7Env
