from __future__ import annotations

from typing import Any, Dict, Iterable


def _as_int(value: Any) -> int:
    return int(bool(value))


def _sum_metric(logs: Iterable[Dict[str, Any]], key: str, alt: str | None = None) -> int:
    total = 0
    for row in logs:
        if key in row:
            total += _as_int(row.get(key, 0))
        elif alt is not None:
            total += _as_int(row.get(alt, 0))
    return total


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def grade_static_scraper(logs) -> float:
    tp = _sum_metric(logs, "TP", "true_positive")
    tn = _sum_metric(logs, "TN", "true_negative")
    total = len(logs)
    if total <= 0:
        return 0.0
    accuracy = (tp + tn) / total
    return _clamp01(accuracy / 0.9)


def grade_distributed_attack(logs) -> float:
    fp = _sum_metric(logs, "FP", "false_positive")
    human_requests = 0
    for row in logs:
        cls = row.get("classification")
        if cls in ("verified_human", "probably_human"):
            human_requests += 1
    if human_requests <= 0:
        return 0.0
    false_positive_rate = fp / human_requests
    return _clamp01(1.0 - (false_positive_rate / 0.05))


def grade_llm_mimicry(logs) -> float:
    tp = _sum_metric(logs, "TP", "true_positive")
    malicious_requests = 0
    for row in logs:
        cls = row.get("classification")
        if cls in ("soft_bot", "hard_bot"):
            malicious_requests += 1
    bot_detection_rate = (tp / malicious_requests) if malicious_requests > 0 else 0.0

    cpu_peak = 0.0
    for row in logs:
        cpu_peak = max(cpu_peak, float(row.get("cpu_load", 0.0)))
    cpu_score = 1.0 - max(0.0, cpu_peak - 0.8) / 0.2

    score = 0.6 * bot_detection_rate + 0.4 * cpu_score
    return _clamp01(score)
