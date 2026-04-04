class RiskEngine:
    def __init__(self):
        # Asymmetric Weights defined by architecture
        self.w_true_positive = 1.0   # Blocked a real bot (Good)
        self.w_false_negative = -3.0 # Allowed a bot through (Bad - CPU spike)
        self.w_false_positive = -10.0 # Blocked a human (Catastrophic - Lost revenue)
        self.w_true_negative = 0.5   # Allowed a human through (Baseline good)

    def compute_risk_score(self, obs) -> bool:
        """
        Determines the GROUND TRUTH if a request is a bot based on the 5D observation.
        This is hidden from the agent.
        """
        # If IP reputation is very low, or it's moving at superhuman speeds
        if obs.ip_reputation < 0.5 or obs.velocity > 0.8:
            return True # It's a bot
            
        # If the payload signature matches known malicious patterns
        if obs.payload_signature > 0.5:
            return True # It's a bot
            
        return False # It's a human

    def compute_reward(self, action: int, is_bot: bool) -> float:
        """
        Calculates the reward using the asymmetric weights.
        Actions: 0 (Allow), 1 (Block), 2 (Deploy Captcha), 3 (Inspect Payload)
        """
        reward = 0.0
        
        if action == 1: # Agent chose to BLOCK
            if is_bot:
                reward = self.w_true_positive
            else:
                reward = self.w_false_positive # Ouch.
                
        elif action == 0: # Agent chose to ALLOW
            if is_bot:
                reward = self.w_false_negative
            else:
                reward = self.w_true_negative
                
        elif action == 2: # DEPLOY CAPTCHA
            # Safe play, but annoys users slightly
            reward = 0.1 if is_bot else -0.5
            
        elif action == 3: # INSPECT PAYLOAD
            # Costs compute time to run the LLM grader
            reward = -0.2 
            
        return reward