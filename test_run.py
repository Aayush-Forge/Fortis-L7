"""Smoke test: random policy on FortisL7Env (Gymnasium API)."""

from __future__ import annotations

import numpy as np

from env import FortisL7Env


def main() -> None:
    env = FortisL7Env(difficulty_level=2, max_episode_steps=50, seed=42)
    obs, info = env.reset(seed=42)
    print("Fortis-L7 smoke test")
    print("obs shape:", obs.shape, "dtype:", obs.dtype)
    print("reset info keys:", sorted(info.keys()))

    for step in range(1, 6):
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n--- step {step} ---")
        print("action:", action, "reward:", reward, "terminated:", terminated)
        print("classification (hidden):", info.get("classification"))
        print("risk_score (hidden):", round(info["risk_score"], 4))
        print("traffic_tier:", info.get("traffic_tier"))
        print("TP/FP/FN:", info.get("true_positive"), info.get("false_positive"), info.get("false_negative"))
        if terminated or truncated:
            break

    print("\nstate() snapshot:", env.state())


if __name__ == "__main__":
    main()
