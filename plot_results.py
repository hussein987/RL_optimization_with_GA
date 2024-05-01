import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_data(log_dir, experiment_names):
    fig = plt.figure(figsize=(10, 7))

    for experiment_name in experiment_names:
        summary_iter = EventAccumulator(f"{log_dir}/{experiment_name}").Reload()

        # Fetch the values for the desired tag
        steps = summary_iter.scalars.Items('rollout/ep_rew_mean')
        # steps = summary_iter.scalars.Items('rollout/ep_len_mean')

        # Extract steps and values
        x = [s.step for s in steps if s.step < 15e4]
        y = [s.value for s in steps if s.step < 15e4]

        plt.plot(x, y, label=experiment_name)
    
    plt.xlabel("Steps")
    plt.ylabel("Rollout Ep Len Mean")
    plt.legend(loc='upper left')
    plt.title("Rollout Ep Reward Mean over Time")
    # plt.title("Rollout Ep Length Mean over Time")
    plt.grid(True)
    # log scale
    # plt.yscale('log')
    # save the figure
    plt.savefig(f"results/rollout_ep_rew_mean_walker.png")
    plt.show()

# log_dir = "tensorboard_logs"  # Path to your logs directory
# experiment_names = ["PPO_bipedalwalker-v3_GA_False", "PPO_bipedalwalker-v3_GA_True"]
# # experiment_names = ["DQN_LunarLander-v2_GA_False", "DQN_LunarLander-v2_GA_True", "PPO_LunarLander-v2_GA_False", "PPO_LunarLander-v2_GA_True"]
# # experiment_names = ["DQN_CartPole-v1_GA_True", "DQN_CartPole-v1_GA_False", "PPO_CartPole-v1_GA_True", "PPO_CartPole-v1_GA_False"]  # Add your experiment names here
# plot_tensorboard_data(log_dir, experiment_names)


# write a function that plot a table of the max reward for each experiment
def plot_table():
    fig = plt.figure(figsize=(10, 7))
    experiment_names = ["PPO_bipedalwalker-v3_GA_False", "PPO_bipedalwalker-v3_GA_True"]
    # experiment_names = ["DQN_LunarLander-v2_GA_False", "DQN_LunarLander-v2_GA_True", "PPO_LunarLander-v2_GA_False", "PPO_LunarLander-v2_GA_True"]
    # experiment_names += ["DQN_CartPole-v1_GA_True", "DQN_CartPole-v1_GA_False", "PPO_CartPole-v1_GA_True", "PPO_CartPole-v1_GA_False"]
    log_dir = "tensorboard_logs"  # Path to your logs directory
    for experiment_name in experiment_names:
        summary_iter = EventAccumulator(f"{log_dir}/{experiment_name}").Reload()

        # Fetch the values for the desired tag
        steps = summary_iter.scalars.Items('rollout/ep_rew_mean')
        # steps = summary_iter.scalars.Items('rollout/ep_len_mean')

        # get the max value
        max_value = max([s.value for s in steps if s.step < 4e5])
        print(f"max value for {experiment_name} is {max_value}")


plot_table()