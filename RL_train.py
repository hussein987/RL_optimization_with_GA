import gym
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


import argparse
import os

def main(env_name, action, with_GA, algo):
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    eval_env = Monitor(eval_env)

    log_dir = "./tensorboard_logs/"

    if algo == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    elif algo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    eval_callback = EvalCallback(eval_env, best_model_save_path=f'./{algo}_best_{env_name}_GA_{with_GA}/',
                                log_path=f'./{algo}_{env_name}_eval/', eval_freq=10000, render=False)

    if action == "train":
        model.learn(total_timesteps=1000000, log_interval=4, callback=eval_callback, tb_log_name=f"{algo}_{env_name}_GA_{with_GA}")
    elif action == "predict":

        if algo == "DQN":
            model = DQN.load(f"{algo}_{env_name}_GA_{with_GA}/best_model.zip")
        elif algo == "PPO":
            model = PPO.load(f"{algo}_{env_name}_GA_{with_GA}/best_model.zip")

        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.1)
            if done:
                obs = env.reset()
    else:
        print("Invalid action. Please use either 'train' or 'predict'.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="DQN CartPole")
    
    env_name = os.environ.get("ENV_NAME", "CartPole-v1")
    action = os.environ.get("ACTION", "train")
    with_GA = os.environ.get("WITH_GA", False)
    algo = os.environ.get("ALGO", "DQN")

    main(env_name, action, with_GA, algo)
