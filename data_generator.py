import random

class RequestGenerator:
    def __init__(self, level: int = 1):
        """
        Initializes the generator with a specific difficulty level (1: Easy, 2: Medium, 3: Hard).
        """
        self.level = level

    def generate_request(self, current_cpu: float) -> dict:
        """
        Generates a 5D observation vector representing a single incoming web request.
        There is a 50/50 chance the request is a legitimate human or a malicious bot.
        """
        is_bot = random.choice([True, False])
        
        if not is_bot:
            # LEGITIMATE TRAFFIC: High reputation, slow velocity, complex navigation
            return {
                "ip_reputation": random.uniform(0.85, 1.0),
                "velocity": random.uniform(0.01, 0.20),
                "nav_path_index": random.uniform(0.60, 0.95),
                "payload_signature": random.uniform(0.0, 0.10),
                "server_cpu_load": current_cpu
            }
            
        # BOT TRAFFIC: Shape depends on the environment's difficulty level
        if self.level == 1:
            # Level 1 (Easy): Static Scrapers
            # High IP Reputation (>0.75) and maximum Velocity (>0.80)
            return {
                "ip_reputation": random.uniform(0.76, 1.0),
                "velocity": random.uniform(0.81, 1.0),
                "nav_path_index": random.uniform(0.0, 0.20),
                "payload_signature": random.uniform(0.0, 0.20),
                "server_cpu_load": current_cpu
            }
            
        elif self.level == 2:
            # Level 2 (Medium): Rotational Proxies
            # Mixed IP Reputation (0.50-0.80) and randomized Velocity
            return {
                "ip_reputation": random.uniform(0.50, 0.80),
                "velocity": random.uniform(0.10, 1.0), 
                "nav_path_index": random.uniform(0.10, 0.40),
                "payload_signature": random.uniform(0.10, 0.30),
                "server_cpu_load": current_cpu
            }
            
        else:
            # Level 3 (Hard): LLM-Mimicry
            # Low IP Reputation (0.10-0.40) and human-like browsing patterns
            return {
                "ip_reputation": random.uniform(0.10, 0.40),
                "velocity": random.uniform(0.05, 0.30), 
                "nav_path_index": random.uniform(0.50, 0.75), # Mimicking humans
                "payload_signature": random.uniform(0.60, 0.95), # Dangerous payload
                "server_cpu_load": current_cpu
            }