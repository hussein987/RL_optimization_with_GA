import gym
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt



# Create the CartPole environment
env = gym.make('CartPole-v1')

# Number of episodes to run
num_episodes = 10

for episode in range(num_episodes):
    print("Runnning")
    # Reset the environment
    observation = env.reset()
    done = False
    step = 0

    while not done:
        # Render the environment
        env.render()

        # Choose a random action
        action = random.choice([0, 1])

        # Take the action
        observation, reward, done, info = env.step(action)[:4]
        print(observation)

        # Increment the step counter
        step += 1

    print(f"Episode {episode+1} finished after {step} steps.")

# Close the environment
env.close()