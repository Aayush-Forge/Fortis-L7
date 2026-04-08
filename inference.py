from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from openai import OpenAI

from env import FortisL7Env
from graders import grade_distributed_attack, grade_llm_mimicry, grade_static_scraper
from tasks import (
    DISTRIBUTED_BOT_DEFENSE,
    LLM_MIMICRY_DETECTION,
    STATIC_SCRAPER_DEFENSE,
)

BENCHMARK_NAME = "fortis-l7-env"


PROMPT_TEMPLATE = """You are an API security agent defending a server.

You receive a request fingerprint:

ip_reputation
velocity_score
entropy_level
navigation_path_index
jitter_value

Each value ranges from 0 to 1.

Interpretation:

High ip_reputation -> suspicious IP
High velocity_score -> aggressive request rate
High entropy_level -> bot-like randomness
High navigation_path_index -> human browsing depth
High jitter_value -> human-like timing

Choose the best mitigation action:

0 = Allow
1 = Throttle
2 = Challenge
3 = Block

Return ONLY the integer action.
"""


def _extract_action(text: str) -> int:
    m = re.search(r"\b([0-3])\b", text or "")
    if not m:
        return 1
    return int(m.group(1))


def _llm_action(client: OpenAI, model_name: str, obs) -> int:
    obs_text = (
        f"ip_reputation={obs[0]:.6f}\n"
        f"velocity_score={obs[1]:.6f}\n"
        f"entropy_level={obs[2]:.6f}\n"
        f"navigation_path_index={obs[3]:.6f}\n"
        f"jitter_value={obs[4]:.6f}\n"
    )
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=4,
        messages=[
            {"role": "system", "content": PROMPT_TEMPLATE},
            {"role": "user", "content": obs_text},
        ],
    )
    content = response.choices[0].message.content or ""
    return _extract_action(content)


def _to_bool_text(value: bool) -> str:
    return "true" if value else "false"


def heuristic_policy(obs) -> int:
    ip_rep, velocity, entropy, nav, jitter = obs

    risk = (
        0.30 * ip_rep
        + 0.25 * velocity
        + 0.20 * entropy
        + 0.15 * (1 - nav)
        + 0.10 * jitter
    )

    if risk > 0.75:
        return 3
    elif risk > 0.55:
        return 2
    elif risk > 0.35:
        return 1
    else:
        return 0


def _select_action(
    risk_score: float,
    obs,
    client: OpenAI,
    model_name: str,
) -> int:
    # Use LLM only in uncertainty zone for stability and lower latency.
    if 0.35 < risk_score < 0.65:
        try:
            return _llm_action(client, model_name, obs)
        except Exception as e:
            print(f"[LLM FALLBACK] API error encountered, using heuristic. Error: {e}")
            return heuristic_policy(obs)
            
    # Outside uncertainty zone, or if we skip LLM
    return heuristic_policy(obs)


def _run_task(task, grader_fn, client: OpenAI, model_name: str) -> float:
    env = FortisL7Env(
        difficulty_level=task.difficulty,
        max_episode_steps=task.steps,
        human_prior=task.human_prior,
    )
    obs, reset_info = env.reset(seed=42 + task.difficulty)
    logs: List[Dict[str, Any]] = []
    rewards: List[float] = []
    success = True
    step_count = 0
    current_risk = float(reset_info.get("risk_score", 0.5))

    print(f"[START] task={task.id} env={BENCHMARK_NAME} model={model_name}")
    for step_idx in range(1, task.steps + 1):
        error_msg = "null"
        done = False
        terminated = False
        truncated = False
        action = 1
        reward = 0.0
        info: Dict[str, Any] = {}
        try:
            action = _select_action(current_risk, obs, client, model_name)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            logs.append(dict(info))
            rewards.append(float(reward))
            current_risk = float(info.get("next_risk_score", info.get("risk_score", current_risk)))
        except Exception as exc:  # noqa: BLE001
            success = False
            done = True
            error_msg = str(exc).replace("\n", " ").strip() or "runtime_error"

        step_count = step_idx
        print(
            "[STEP] "
            f"step={step_idx} action={action} reward={float(reward):.2f} "
            f"done={_to_bool_text(done)} error={error_msg}"
        )
        if terminated or truncated:
            break
        if error_msg != "null":
            break

    score = float(grader_fn(logs))
    rewards_csv = ",".join(f"{r:.2f}" for r in rewards)
    print(
        "[END] "
        f"success={_to_bool_text(success)} steps={step_count} score={max(0.0, min(score, 1.0)):.2f} rewards={rewards_csv}"
    )
    return score


def main() -> None:
    # Required env vars for OpenEnv runtime
    api_base_url = os.getenv("API_BASE_URL","https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    hf_token = os.getenv("HF_TOKEN")

    # Required initialization pattern from the submission prompt.
    client = OpenAI(base_url=os.getenv("API_BASE_URL"), api_key=hf_token)

    if not api_base_url:
        raise RuntimeError("API_BASE_URL must be set.")
    if not hf_token:
        raise RuntimeError("HF_TOKEN must be set.")

    _run_task(STATIC_SCRAPER_DEFENSE, grade_static_scraper, client, model_name)
    _run_task(DISTRIBUTED_BOT_DEFENSE, grade_distributed_attack, client, model_name)
    _run_task(LLM_MIMICRY_DETECTION, grade_llm_mimicry, client, model_name)


if __name__ == "__main__":
    main()