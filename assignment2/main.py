import numpy as np
import mlrose.mlrose_hiive as mlrose
from mlrose.mlrose_hiive.generators.four_peaks_generator import FourPeaksGenerator
from tqdm import tqdm
from enum import Enum
import torch
import os


class RandomOptimizationMethods(Enum):
    SIMULATED_ANNEALING = 1
    RANDOMIZED_HILL_CLIMBING = 2
    GENETIC_ALGORITHM = 3
    MIMIC = 4


def solve(problem, method=RandomOptimizationMethods.SIMULATED_ANNEALING, n_seeds=100, init_seed=123456789):
    ''''''
    np.random.seed(seed=init_seed)
    seeds = [np.random.randint(-10000, 10000) for i in range(n_seeds)]

    fitness_results = []
    iterations = []

    best_fitness = None
    best_state = None
    for seed in tqdm(seeds):
        if method == RandomOptimizationMethods.SIMULATED_ANNEALING:
            state, fitness, curve = mlrose.simulated_annealing(
                problem=problem, schedule=mlrose.GeomDecay(), random_state=seed, curve=True)
        elif method == RandomOptimizationMethods.RANDOMIZED_HILL_CLIMBING:
            state, fitness, curve = mlrose.random_hill_climb(
                problem=problem, random_state=seed, restarts=np.random.randint(0, 10), curve=True)
        elif method == RandomOptimizationMethods.GENETIC_ALGORITHM:
            state, fitness, curve = mlrose.genetic_alg(
                problem=problem, pop_size=100, mutation_prob=np.random.rand(), max_iters=100, random_state=seed, curve=True)
        elif method == RandomOptimizationMethods.MIMIC:
            state, fitness, curve = mlrose.mimic(
                problem=problem, pop_size=100, max_iters=100, keep_pct=np.random.rand(), random_state=seed, curve=True)
        else:
            raise NotImplementedError()
        if best_fitness is None or fitness > best_fitness:
            best_fitness = fitness
            best_state = state
        fitness_results.append(fitness)
        iterations.append(curve.shape[0])
    return best_state, best_fitness, np.mean(fitness_results), np.mean(iterations)


def main():
    '''
    '''
    # problem = mlrose.generators.KnapsackGenerator().generate(seed=1234)

    n_seeds = 100

    # fitness = mlrose.OneMax()
    # problem = mlrose.DiscreteOpt(
    #     length=10, fitness_fn=fitness, maximize=True, max_val=1000)

    # problem = mlrose.generators.KnapsackGenerator().generate(seed=1234)
    # problem = mlrose.generators.FlipFlopGenerator().generate(seed=1234, size=50)
    # problem = mlrose.generators.TSPGenerator().generate(seed=1234, number_of_cities=10)
    # problem = FourPeaksGenerator().generate(seed=1234, size=10)

    # for method in [RandomOptimizationMethods.SIMULATED_ANNEALING,
    #                RandomOptimizationMethods.RANDOMIZED_HILL_CLIMBING,
    #                RandomOptimizationMethods.GENETIC_ALGORITHM,
    #                RandomOptimizationMethods.MIMIC]:
    #     best_state, best_fitness, avg_fitness, avg_iteration = solve(
    #         problem, method, n_seeds=n_seeds)
    #     print(f"{method.name} - {best_state}, {best_fitness}, {avg_iteration}")

    model = torch.load(os.path.join('checkpoints', 'drybean_best_model.pt'))


if __name__ == '__main__':
    main()
