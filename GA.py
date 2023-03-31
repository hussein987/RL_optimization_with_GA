import numpy as np
from typing import Dict, List, Tuple

class GA:
    def __init__(self, params: Dict[str, List], fitness_fn: Callable[[Dict[str, any]], float], tournament_size: int = 2, mutation_prob: float = 0.1, pop_size: int = 10):
        """
        :param params: a dict with the name of the parameter as the key
        and a list of parameter values as the value for that key
        :param fitness_fn: a function that takes a dictionary of hyperparameters as input and returns a fitness score
        :param tournament_size: the size of the tournament for selection (default: 2)
        :param mutation_prob: the probability of mutation for each parameter (default: 0.1)
        :param pop_size: the size of the population (default: 10)
        """
        self.params = params
        self.fitness_fn = fitness_fn
        self.tournament_size = tournament_size
        self.mutation_prob = mutation_prob
        self.pop_size = pop_size

    def init_population(self) -> List[Dict[str, any]]:
        """
        Initializes the population with random hyperparameter values.

        Returns:
        A list of dictionaries, where each dictionary contains hyperparameter
        values for an individual in the population.
        """
        population = []
        for i in range(self.pop_size):
            s = {}
            for param in self.params:
                s[param] = np.random.choice(self.params[param])
            population.append(s)
        return population

    def do_crossover(self, s1: Dict[str, any], s2: Dict[str, any], m: int) -> Tuple[Dict[str, any], Dict[str, any]]:
        """
        Performs crossover between two individuals at a given crossover point.

        Args:
        s1: A dictionary containing hyperparameter values for the first individual.
        s2: A dictionary containing hyperparameter values for the second individual.
        m: The crossover point, i.e., the index of the parameter where crossover occurs.

        Returns:
        A tuple containing two dictionaries, each representing an offspring
        resulting from the crossover operation.
        """
        if m >= len(self.params):
            raise ValueError("Crossover point exceeds the number of parameters.")

        c1, c2 = {}, {}
        for idx, parameter in enumerate(self.params):
            if idx < m:
                c1[parameter] = s1[parameter]
                c2[parameter] = s2[parameter]
            else:
                c1[parameter] = s2[parameter]
                c2[parameter] = s1[parameter]
        return c1, c2

    def do_mutation(self, s: Dict[str, any], m: int = None) -> Dict[str, any]:
        """
        Mutates an individual in the population by randomly changing one of its hyperparameters.
        The hyperparameter to be mutated is randomly selected from the list of hyperparameters.

        :param s: a dictionary representing an individual in the population
        :param m: the index of the hyperparameter to be mutated (optional, if not specified, a hyperparameter is chosen at random)
        :return: a new dictionary representing the mutated individual
        """
        if m is None:
            m = np.random.choice(list(self.params.keys()))

        if m not in self.params:
            raise ValueError("Invalid parameter name for mutation.")

        s1 = s.copy()
        s1[m] = np.random.choice(self.params[m])
        return s1

    def get_elite(self, gen: List[Dict[str, any]], fitness: List[float], k: int) -> List[Dict[str, any]]:
        """
        Returns the top k individuals in the population based on their fitness scores.
        The population and corresponding fitness scores are sorted in descending order of fitness.

        :param gen: a list of dictionaries representing individuals in the population
        :param fitness: a list of fitness scores corresponding to the individuals in the population
        :param k: the number of elite individuals to select
        :return: a list of the top k individuals in the population based on fitness score
        """
        if len(gen) != len(fitness):
            raise ValueError("Population size and fitness values do not match.")

        if k > len(gen):
            raise ValueError("Number of elites requested exceeds the population size.")

        temp_list = [(gen[i], fitness[i]) for i in range(len(gen))]
        temp_list = sorted(temp_list, key=lambda x: x[1], reverse=True)
        return [temp[0] for temp in temp_list][:k]
