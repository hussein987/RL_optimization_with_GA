# load this model /Users/husseinyounes/University/Advanced_ML/RL_optimization_with_GA/dqn_cartpole_eval/best_model.zip and evaluate it

import gym
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

eval_env = gym.make("CartPole-v1")

print("Loading model...")
# load /Users/husseinyounes/University/Advanced_ML/RL_optimization_with_GA/dqn_cartpole_eval/best_model.zip
model = DQN.load("/Users/husseinyounes/University/Advanced_ML/RL_optimization_with_GA/dqn_cartpole_eval/best_model.zip")
print("Model loaded")
print(model)

obs = eval_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    eval_env.render()
    time.sleep(0.01)
    if done:
        obs = eval_env.reset()