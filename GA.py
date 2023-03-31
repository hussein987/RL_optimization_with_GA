import numpy as np
from typing import Dict, List, Tuple

class GA:
    def __init__(self, params: Dict[str, List]):
        '''
        :param params: a dict with the name of the parameter as the key
        and a list of parameter values as the value for that key
        '''
        self.params = params

    def do_crossover(self, s1: Dict[str, any], s2: Dict[str, any], m: int) -> Tuple[Dict[str, any], Dict[str, any]]:
        """
        s1 and s2 are dictionaries of the hyper-parameters
        m is the point of cross over
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
        if m is None:
            m = np.random.choice(list(self.params.keys()))

        if m not in self.params:
            raise ValueError("Invalid parameter name for mutation.")

        s1 = s.copy()
        s1[m] = np.random.choice(self.params[m])
        return s1

    def get_elite(self, gen: List[Dict[str, any]], fitness: List[float], k: int) -> List[Dict[str, any]]:
        if len(gen) != len(fitness):
            raise ValueError("Population size and fitness values do not match.")

        if k > len(gen):
            raise ValueError("Number of elites requested exceeds the population size.")

        temp_list = [(gen[i], fitness[i]) for i in range(len(gen))]
        temp_list = sorted(temp_list, key=lambda x: x[1], reverse=True)
        return [temp[0] for temp in temp_list][:k]
