import gym
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor


with_GA = False
env_name = "CartPole-v1"

env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")
eval_env = Monitor(eval_env)


log_dir = "./dqn_tensorboard/"

# model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
# PPO model
from stable_baselines3 import PPO
# model with the default hyperparameters
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

eval_callback = EvalCallback(eval_env, best_model_save_path=f'./best_{env_name}_GA_{with_GA}/',
                             log_path='./dqn_cartpole_eval/', eval_freq=10000, render = True)


# Pass the TensorBoard callback to the learn method
model.learn(total_timesteps=1000000, log_interval=4, callback=eval_callback, tb_log_name=f"PPO_{env_name}_GA_{with_GA}")
model.save("dqn_cartpole")

del model  # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.1)
    if done:
        obs = env.reset()
