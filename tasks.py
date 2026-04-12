from __future__ import annotations

from pydantic import BaseModel, Field


def clamp01(value: float) -> float:
    """Strict open interval (0, 1) — never returns 0.0 or 1.0."""
    return max(0.001, min(0.999, float(value)))


class TaskDefinition(BaseModel):
    id: str
    difficulty: int = Field(ge=1, le=3)
    steps: int = Field(gt=0)
    human_prior: float = Field(ge=0.0, le=1.0)
    description: str


STATIC_SCRAPER_DEFENSE = TaskDefinition(
    id="static_scraper_defense",
    difficulty=1,
    steps=50,
    human_prior=0.2,
    description="Mitigate easy bots with high accuracy.",
)

DISTRIBUTED_BOT_DEFENSE = TaskDefinition(
    id="distributed_bot_defense",
    difficulty=2,
    steps=75,
    human_prior=0.4,
    description="Maintain low false positives under mixed traffic.",
)

LLM_MIMICRY_DETECTION = TaskDefinition(
    id="llm_mimicry_detection",
    difficulty=3,
    steps=100,
    human_prior=0.5,
    description="Detect sophisticated bots while keeping CPU load stable.",
)

TASKS = {
    STATIC_SCRAPER_DEFENSE.id: STATIC_SCRAPER_DEFENSE,
    DISTRIBUTED_BOT_DEFENSE.id: DISTRIBUTED_BOT_DEFENSE,
    LLM_MIMICRY_DETECTION.id: LLM_MIMICRY_DETECTION,
}