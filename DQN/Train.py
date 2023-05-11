import copy

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from GA import GA

Game = "CartPole-v1"
# Game = "LunarLander-v2"


def train(env, learning_rate=1e-4, gamma=0.99, total_timesteps=1e4, progress_bar=False, **kwargs):
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=0,
        exploration_final_eps=0.1,
        target_update_interval=250,
    )
    model.learn(total_timesteps=total_timesteps, log_interval=4, progress_bar=progress_bar)

    return model


def evaluate(model, env, render=False):
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, render=render)
    return mean_reward


params = {
    'learning_rate': np.linspace(1e-5, 1e-3, 100),
    'gamma': np.linspace(0.7, 0.99, 10),
}
ga = GA(params=params)
pop = ga.init_population()
num_iterations = 5
best_fit = -1000000000000
best_individual = []
best_model = None
for iteration in range(num_iterations):
    print(f'Starting iteration {iteration}')

    print('Population training')
    fitness = np.zeros_like(pop)
    for idx, individual in tqdm(enumerate(pop)):
        if individual['trained']:
            fitness[idx] = individual['fitness']
        else:
            env = gym.make("LunarLander-v2")
            model = train(env, **individual)
            fitness[idx] = evaluate(model, env)
            individual['trained'] = True
            individual['fitness'] = fitness[idx]

        if fitness[idx] >= best_fit:
            best_fit = fitness[idx]
            best_model = copy.deepcopy(model)
            best_individual = copy.deepcopy(individual)

    best_model.save(Game + '_best_ppo_model')
    print(f'{best_individual = }, {best_fit =}')

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

print('Training the best model fully')
model = train(env, total_timesteps=int(5e4), progress_bar=True, **best_individual)
model.save(Game + '_best_ppo_model_full')
evaluate(model, env, render=True)