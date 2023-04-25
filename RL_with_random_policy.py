import gym
import random
import matplotlib.pyplot as plt

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")  # rgb_array")

# Number of episodes to run
num_episodes = 10

for episode in range(num_episodes):
    # Reset the environment
    observation = env.reset()
    done = False
    step = 0

    while not done:
        # Render the environment
        env.render()
        import time

        time.sleep(0.02)

        # Choose a random action
        action = random.choice([0, 1])

        # Take the action
        observation, reward, done, info = env.step(action)[:4]

        # Increment the step counter
        step += 1

    print(f"Episode {episode+1} finished after {step} steps.")

# Close the environment
env.close()
