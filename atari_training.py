import gym
from stable_baselines3.common.atari_wrappers import wrap_deepmind
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

import time

# train breakout atari
env = gym.make('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env, frame_stack=True, scale=True)
env = VecFrameStack(env, n_stack=4)

# use DQN to train the agent
from stable_baselines3 import DQN
print("Training DQN agent...")
model = DQN('CnnPolicy', env, verbose=1, tensorboard_log="./dqn_breakout_tensorboard/")

print("Evaluating DQN agent...")

eval_callback = EvalCallback(env, best_model_save_path='./dqn_breakout_eval/',
                             log_path='./dqn_breakout_eval/', eval_freq=10000, render = True)

model.learn(total_timesteps=1000000, log_interval=4, callback=eval_callback, tb_log_name=f"DQN_BreakoutNoFrameskip-v4")
model.save("dqn_breakout")

del model  # remove to demonstrate saving and loading

model = DQN.load("dqn_breakout")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.1)
    if done:
        obs = env.reset()