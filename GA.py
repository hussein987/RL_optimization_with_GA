import numpy as np


class GA:
    def __init__(self) -> None:
        pass

    def do_crossover(self, s1, s2, m):
        """
        s1 and s2 are list of the hyoer parameters
        m is the point of cross over
        """
        c1 = s1[:m].extend(s2[m:])
        c2 = s2[:m].extend(s1[m:])
        return c1, c2

    def do_mutation(self, s, m):
        s1 = s.copy()
        s1[m] += np.random.choice([-0.1, 0.1])
        return s1

    def compute_fitness(s):
        return 0

    def get_elite(self, gen, k):
        gen = sorted(gen, key=lambda s: self.compute_fitness(s))
        return gen[:k]
