from env import FortisL7Env
import random

# Initialize the environment on Hard Mode (Level 3)
env = FortisL7Env(difficulty_level=3)

# Reset the environment to get the first state
initial_state = env.reset()
print("🔥 FORTIS L7 SIMULATION START 🔥")
print(f"Initial State: {initial_state}\n")

# Run 5 random actions to see how the engine responds
for step in range(1, 6):
    # Agent picks a random action (0: Allow, 1: Block, 2: Captcha, 3: Inspect)
    action = random.choice([0, 1, 2, 3])
    
    # Step the environment forward
    next_obs, reward, done, info = env.step(action)
    
    print(f"--- Step {step} ---")
    print(f"Agent Action: {action}")
    print(f"Ground Truth (Was Bot?): {info['ground_truth_was_bot']}")
    print(f"Reward Received: {reward}")
    print(f"Current Server CPU: {info['current_cpu']:.2f}")
    
    if done:
        print("\n💥 SERVER CRASHED OR MAX STEPS REACHED 💥")
        break