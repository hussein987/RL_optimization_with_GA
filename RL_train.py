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
    eval_env = gym.make(env_name, render_mode="human")
    eval_env = Monitor(eval_env)

    log_dir = "./results/logs/tensorboard_logs_new/"

    if algo == "DQN":
        if args.with_GA:
            model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        else:
            model = DQN(
                "MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=1e-2
            )
    elif algo == "PPO":
        if args.with_GA:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=1e-3,
                gamma=0.8,
                n_steps=32,
                ent_coef=0.01,
                gae_lambda=0.8,
            )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./{algo}_best_{env_name}_GA_{str(with_GA)}/",
        log_path=f"./{algo}_{env_name}_eval/",
        eval_freq=10000,
        render=True,
    )

    if action == "train":
        model.learn(
            total_timesteps=1000000,
            log_interval=4,
            callback=eval_callback,
            tb_log_name=f"{algo}_{env_name}_GA_{str(with_GA)}",
        )
    elif action == "predict":

        if algo == "DQN":
            # model = DQN.load(f"{algo}_{env_name}_GA_{with_GA}/best_model.zip")
            model = DQN(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=1e-5,
                gamma=0.8,
                batch_size=32,
                exploration_fraction=0.9,
            )
        elif algo == "PPO":
            # model = PPO.load(f"{algo}_{env_name}_GA_{with_GA}/best_model.zip")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=1e-5,
                gamma=0.8,
                n_steps=32,
                ent_coef=0.01,
                gae_lambda=0.8,
            )

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
    print("Starting training")
    parser = argparse.ArgumentParser(description="DQN CartPole")
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument("--action", type=str, default="train", help="Action to perform")
    parser.add_argument("--with_GA", type=bool, default=False, help="Whether to use GA")
    parser.add_argument("--algo", type=str, default="DQN", help="Algorithm to use")

    # env_name = os.environ.get("ENV_NAME", "CartPole-v1")
    # action = os.environ.get("ACTION", "train")
    # with_GA = os.environ.get("WITH_GA", False)
    # algo = os.environ.get("ALGO", "DQN")

    args = parser.parse_args()
    print(args)
    main(args.env_name, args.action, args.with_GA, args.algo)

    # python RL_train.py --env_name CartPole-v1 --action train --with_GA False --algo DQN
