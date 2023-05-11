# RL_optimization_with_GA

## Genetic Algorithms

We will use GA to fine tune the model hyperparameters parameters.
There are different implementations like DEAP, Pyevolve and Tpot that  we could use or we can just implement our own.

We have implemented Proximal Policy Optimization (PPO) and QDN algorithms for OpenAI gym environment using PyTorch.

Requirements
To run this code, you will need the following packages:

- gym
- PyTorch
- NumPy
- stable_baselines3

- You can install them using pip:
```
pip install gym torch numpy stable_baselines3
```

Usage
To run the code, go to the directory of the model that you want to train (PPO or DQN),
and execute the Train.py script:

```
python train.py
```
You can edit the parameters given to the GA class to edit the hyperparameters to be optimized.

The script will train a PPO/DQN agent for 10000 episodes of the environment
and  print the total score obtained in every generation. The trained network
will be saved in a file next to the code.

