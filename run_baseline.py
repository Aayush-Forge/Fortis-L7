import random
from env import FortisL7Env

def run_evaluation(difficulty_level: int, episodes: int = 5):
    """
    Runs a baseline random agent through the environment for a set number of episodes.
    Calculates a final normalized score between 0.0 and 1.0 for the judges.
    """
    print(f"\n🛡️  Running Baseline Agent on Task Level {difficulty_level}...")
    env = FortisL7Env(difficulty_level=difficulty_level)
    
    total_scores = []
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        step_count = 0
        
        while not done:
            # Baseline Agent: Randomly selects an action (0: Allow, 1: Block, 2: Captcha, 3: Inspect)
            action = random.choice([0, 1, 2, 3])
            
            # Environment processes the action
            next_obs, reward, done, info = env.step(action)
            step_count += 1
        
        # --- OPENENV SCORING LOGIC (0.0 to 1.0) ---
        # A perfect 1.0 means the agent survived all 100 steps while keeping the CPU cool.
        survival_rate = step_count / env.max_steps
        cpu_penalty = info['current_cpu'] # CPU ranges from 0.10 to 1.0
        
        # Calculate final metric
        final_score = max(0.0, survival_rate - (cpu_penalty * 0.2))
        
        # If the server crashed (CPU hit 1.0), the run is an automatic failure
        if info['current_cpu'] >= 1.0:
            final_score = 0.0 
            
        total_scores.append(final_score)
        print(f"  Episode {episode + 1} | Steps Survived: {step_count:03d}/100 | Final Score: {final_score:.2f}")

    avg_score = sum(total_scores) / episodes
    print(f"✅ Level {difficulty_level} Average Baseline Score: {avg_score:.2f} / 1.0\n")

if __name__ == "__main__":
    print("==================================================")
    print("   FORTIS L7: AUTOMATED OPENENV EVALUATION")
    print("==================================================")
    
    # Run the agent against all 3 difficulty tiers to satisfy the hackathon requirement
    run_evaluation(difficulty_level=1)
    run_evaluation(difficulty_level=2)
    run_evaluation(difficulty_level=3)
    
    print("==================================================")
    print("🏁 Baseline evaluation complete. Ready for submission.")