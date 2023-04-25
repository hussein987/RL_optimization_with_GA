import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

def rollout(policy, env, max_steps=200):
    states, actions, rewards = [], [], []
    state = env.reset()

    for _ in range(max_steps):
        print(state)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        action_probs = policy(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        
        new_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        if done:
            break
        
        state = new_state

    return np.array(states), np.array(actions), np.array(rewards)


def calculate_returns(rewards, gamma=0.99):
    returns = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns

def train_ppo(policy, optimizer, states, actions, returns, epochs=10, eps_clip=0.2):
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    returns_tensor = torch.FloatTensor(returns)

    for _ in range(epochs):
        action_probs = policy(states_tensor)
        action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            old_action_probs = action_probs.clone()

        advantages = returns_tensor - action_probs

        for state, action, advantage, old_action_prob in zip(states_tensor, actions_tensor, advantages, old_action_probs):
            optimizer.zero_grad()

            cur_action_prob = policy(state.unsqueeze(0)).gather(1, action.unsqueeze(0)).squeeze(1)

            ratio = cur_action_prob / old_action_prob
            clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)

            loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

            loss.backward()
            optimizer.step()

def main():
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    num_episodes = 10

    for episode in range(num_episodes):
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
            observation, reward, done, info = env.step(action)

            # Increment the step counter
            step += 1

        print(f"Episode {episode+1} finished after {step} steps.")

    # Close the environment
    env.close()

    return
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = Policy(state_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    max_episodes = 500
    max_steps = 200

    for episode in range(max_episodes):
        states, actions, rewards = rollout(policy, env, max_steps)
        returns = calculate_returns(rewards)

        train_ppo(policy, optimizer, states, actions, returns)

        if episode % 10 == 0:
            print(f"Episode: {episode}, Reward: {sum(rewards)}")

if __name__ == "__main__":
    main()
