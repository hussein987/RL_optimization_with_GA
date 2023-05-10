import gym
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

import time


env_name = "LunarLander-v2"
env = gym.make(env_name)

# use DQN to train the agent
from stable_baselines3 import DQN
print("Training DQN agent...")

log_dir = "./LunarLander-v2_tensorboard/"
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

print("Evaluating DQN agent...")

eval_callback = EvalCallback(env, best_model_save_path='./LunarLander-v2_eval_1/',
                             log_path='./LunarLander-v2_eval_1/', eval_freq=10000, render = True)

model.learn(total_timesteps=1000000, log_interval=4, callback=eval_callback, tb_log_name=f"DQN_{env_name}")
model.save("LunarLander-v2")

del model  # remove to demonstrate saving and loading

model = DQN.load("LunarLander-v2")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.1)
    if done:
        obs = env.reset()