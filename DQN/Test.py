import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env


n_tests = 1
Game = "CartPole-v1"
Game = "LunarLander-v2"

# Parallel environments
env = make_vec_env(Game, n_envs=n_tests)

model = DQN.load(Game + "_best_ppo_model_fUll")


# obs = env.reset()
# done = False
# Total_rewards = 0
# count = 0
# while not done:
#     action, _states = model.predict(obs.reshape((8,)))
#     obs, rewards, dones, info = env.step(action)
#     Total_rewards += rewards
#     print(dones)
#     done = True
#     for i in dones:
#         done = done and i
#     count += 1
#     print('\r', count, Total_rewards, end='')
#
#     env.render()
while True:

    obs = env.reset()
    done = False
    total_rewards = 0
    count = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        total_rewards += rewards

        count += 1
        print('\r', count, total_rewards, end='')
        env.render()
    print()
