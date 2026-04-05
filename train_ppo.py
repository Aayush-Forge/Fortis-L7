"""
Train PPO on FortisL7Env and evaluate bot/human outcome rates.
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from env import FortisL7Env


def make_env(difficulty: int, rank: int) -> FortisL7Env:
    return FortisL7Env(
        difficulty_level=difficulty,
        max_episode_steps=256,
        human_prior=0.35,
        seed=1000 + rank,
        use_cpu_dynamics=True,
    )


def rollout_eval(model: PPO, difficulty: int, episodes: int = 20, seed: int = 123) -> None:
    env = FortisL7Env(
        difficulty_level=difficulty,
        max_episode_steps=256,
        human_prior=0.35,
        seed=seed,
        use_cpu_dynamics=True,
    )
    tp = fp = fn = tn = 0
    total_r = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        trunc = False
        ep_r = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(int(action))
            ep_r += float(r)
            if info.get("true_positive"):
                tp += 1
            if info.get("false_positive"):
                fp += 1
            if info.get("false_negative"):
                fn += 1
            if info.get("true_negative"):
                tn += 1
        total_r += ep_r

    steps = tp + fp + fn + tn
    print("\n--- Eval rollout ---")
    print(f"Episodes: {episodes}  total steps: {steps}  mean return: {total_r / episodes:.2f}")
    if steps:
        print(
            f"TP rate: {tp / steps:.3f}  FP rate: {fp / steps:.3f}  "
            f"FN rate: {fn / steps:.3f}  TN rate: {tn / steps:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save", type=str, default="fortis_ppo")
    args = parser.parse_args()

    def factory(rank: int) -> FortisL7Env:
        return make_env(args.difficulty, rank)

    vec = make_vec_env(factory, n_envs=args.n_envs, vec_env_cls=DummyVecEnv)
    vec = VecMonitor(vec)

    model = PPO(
        "MlpPolicy",
        vec,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        tensorboard_log="./tb_fortis/",
    )
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
    except ImportError:
        model.learn(total_timesteps=args.timesteps, progress_bar=False)

    path = args.save + ".zip"
    model.save(path)
    print(f"Saved model to {path}")

    rollout_eval(model, args.difficulty)


if __name__ == "__main__":
    main()
