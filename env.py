from pydantic import BaseModel, Field
from openenv_core import Environment
from risk_engine import RiskEngine
from data_generator import RequestGenerator

# 1. Pydantic Model for Strict Data Validation
class Observation(BaseModel):
    ip_reputation: float = Field(ge=0.0, le=1.0)
    velocity: float = Field(ge=0.0, le=1.0)
    nav_path_index: float = Field(ge=0.0, le=1.0)
    payload_signature: float = Field(ge=0.0, le=1.0)
    server_cpu_load: float = Field(ge=0.0, le=1.0)

class FortisL7Env(Environment):
    def __init__(self, difficulty_level: int = 1):
        super().__init__()
        self.difficulty = difficulty_level
        # Initialize our custom modules
        self.generator = RequestGenerator(level=self.difficulty)
        self.risk_engine = RiskEngine()
        
        # Internal state tracking
        self.current_obs = None
        self.cpu_load = 0.10
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        """Initialize the environment and return the first observation vector."""
        self.current_step = 0
        self.cpu_load = 0.10
        
        # Pull the first synthetic request from our generator
        raw_data = self.generator.generate_request(self.cpu_load)
        
        # Validate using Pydantic before passing to the agent
        self.current_obs = Observation(**raw_data)
        
        return self.current_obs.model_dump()

    def step(self, action: int):
        """Receive agent action (0-3), advance state, and return (next_obs, reward, done, info)."""
        self.current_step += 1
        
        # Get Ground Truth (Hidden from agent)
        is_bot = self.risk_engine.compute_risk_score(self.current_obs)
        
        # Calculate Asymmetric Reward
        reward = self.risk_engine.compute_reward(action, is_bot)
        
        # Simulate Server Physics (Allowing a bot spikes the CPU)
        if action == 0 and is_bot:
            self.cpu_load = min(1.0, self.cpu_load + 0.15)
        elif action == 0 and not is_bot:
            # Processing normal traffic smoothly cools the CPU down slightly
            self.cpu_load = max(0.10, self.cpu_load - 0.05)
            
        # Check termination condition (Server Crash or Time limit reached)
        done = bool(self.cpu_load >= 1.0 or self.current_step >= self.max_steps)
        
        # Generate the next request for the queue
        raw_data = self.generator.generate_request(self.cpu_load)
        self.current_obs = Observation(**raw_data)
        
        info = {
            "ground_truth_was_bot": is_bot,
            "current_cpu": self.cpu_load,
            "agent_action": action
        }
        
        return self.current_obs.model_dump(), reward, done, info

    def state(self):
        """Provide a read-only snapshot of the internal state for the Meta judges/logging."""
        return {
            "current_step": self.current_step,
            "cpu_status": self.cpu_load,
            "latest_request": self.current_obs.model_dump() if self.current_obs else None
        }