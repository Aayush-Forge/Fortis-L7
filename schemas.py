from pydantic import BaseModel, Field


class ObservationModel(BaseModel):
    ip_reputation: float = Field(ge=0.0, le=1.0)
    velocity_score: float = Field(ge=0.0, le=1.0)
    entropy_level: float = Field(ge=0.0, le=1.0)
    navigation_path_index: float = Field(ge=0.0, le=1.0)
    jitter_value: float = Field(ge=0.0, le=1.0)


class ActionModel(BaseModel):
    action: int = Field(ge=0, le=3)


class RewardModel(BaseModel):
    reward: float
    risk_score: float
    classification: str
    cpu_load: float
