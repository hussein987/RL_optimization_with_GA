import gym
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from GA import GA

Game = "CartPole-v1"
Game = "LunarLander-v2"
## learning_rate: Union[float, Schedule] = 3e-4,
# n_steps: int = 2048,
#         batch_size: int = 64,
## n_epochs: int = 10,
## gamma: float = 0.99,
# gae_lambda: float = 0.95,
# clip_range: Union[float, Schedule] = 0.2,
#         clip_range_vf: Union[None, float, Schedule] = None,
# ent_coef: float = 0.0,
# vf_coef: float = 0.5,
# max_grad_norm: float = 0.5,
# use_sde: bool = False,
# sde_sample_freq: int = -1,
# target_kl: Optional[float] = None,
# stats_window_size: int = 100,
#         tensorboard_log: Optional[str] = None,
#         policy_kwargs: Optional[Dict[str, Any]] = None,
#         verbose: int = 0,
#         seed: Optional[int] = None,
#         device: Union[th.device, str] = "auto",
#         _init_setup_model: bool = True,

def train(env, learning_rate=3e-4, gamma=0.99, n_epochs=10):
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_epochs=n_epochs,
        n_steps=256,
        verbose=0,
    )
    model.learn(total_timesteps=4096*4)

    return model


def evaluate(model, env):
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1, render=False)
    # obs = env.reset()
    # done = False
    # total_rewards = 0
    # count = 0
    # while not done and count < 2000:
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     total_rewards += rewards
    #
    #     count += 1

    return mean_reward


params = {
    'learning_rate': np.linspace(1e-5, 1e-1, 100),
    'gamma': np.linspace(0.5, 0.99, 100),
    # 'n_epochs': np.linspace(10, 50, 10, dtype=int),
}
ga = GA(params=params)
pop = ga.init_population()
num_iterations = 2
best_fit = -1000000000000
for iteration in range(num_iterations):
    print(f'Starting iteration {iteration}')

    print('Population training')
    fitness = np.zeros_like(pop)
    for idx, individual in tqdm(enumerate(pop)):
        env = gym.make("LunarLander-v2")
        model = train(env, **individual)
        fitness[idx] = evaluate(model, env)
        if fitness[idx] >= best_fit:
            best_fit = fitness[idx]
            model.save(Game + '_best_ppo_model')

    # Elitesm
    pop, fitness = ga.get_elite(pop=pop, fitness=fitness)

    # Crossover
    for i in range(ga.tournament_size):
        c1, c2 = ga.do_crossover(pop[i], np.random.choice(pop))
        pop.append(c1)
        pop.append(c2)

    # Mutation
    for i in range(len(pop)):
        pop[i] = ga.do_mutation(s=pop[i])

    print(f'{pop[0] = }, {fitness[0] = }\n\n')
