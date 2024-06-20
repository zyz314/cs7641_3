import numpy as np
import mlrose.mlrose_hiive as mlrose
from mlrose.mlrose_hiive.generators.four_peaks_generator import FourPeaksGenerator
from tqdm import tqdm
from enum import Enum
import torch
import os
from pyperch.neural import RHCModule
import itertools
import matplotlib.pyplot as plt


class RandomOptimizationMethods(Enum):
    SIMULATED_ANNEALING = 1
    RANDOMIZED_HILL_CLIMBING = 2
    GENETIC_ALGORITHM = 3
    MIMIC = 4


def solve(problem, method=RandomOptimizationMethods.SIMULATED_ANNEALING, n_seeds=100, init_seed=123456789, max_iters=100):
    ''''''
    np.random.seed(seed=init_seed)
    seeds = [np.random.randint(-10000, 10000) for i in range(n_seeds)]

    fitness_results = []
    iterations = []
    curves = []

    best_fitness = None
    best_state = None
    for seed in tqdm(seeds):
        if method == RandomOptimizationMethods.SIMULATED_ANNEALING:
            state, fitness, curve = mlrose.simulated_annealing(
                problem=problem, schedule=mlrose.GeomDecay(), random_state=seed, max_iters=max_iters, curve=True)
        elif method == RandomOptimizationMethods.RANDOMIZED_HILL_CLIMBING:
            state, fitness, curve = mlrose.random_hill_climb(
                problem=problem, random_state=seed, restarts=np.random.randint(0, 10), max_iters=max_iters, curve=True)
        elif method == RandomOptimizationMethods.GENETIC_ALGORITHM:
            state, fitness, curve = mlrose.genetic_alg(
                problem=problem, pop_size=100, mutation_prob=np.random.rand(), max_iters=max_iters, random_state=seed, curve=True)
        elif method == RandomOptimizationMethods.MIMIC:
            state, fitness, curve = mlrose.mimic(
                problem=problem, pop_size=100, max_iters=max_iters, keep_pct=np.random.rand(), random_state=seed, curve=True)
        else:
            raise NotImplementedError()
        if best_fitness is None or fitness > best_fitness:
            best_fitness = fitness
            best_state = state
        fitness_results.append(fitness)
        iterations.append(curve.shape[0])
        curves.append(curve[:, 0])
    return best_state, best_fitness, np.mean(fitness_results), np.mean(iterations), curves


def main():
    '''
    '''
    # problem = mlrose.generators.KnapsackGenerator().generate(seed=1234)

    n_seeds = 10

    fitness = mlrose.OneMax()
    problem = mlrose.DiscreteOpt(
        length=10000, fitness_fn=fitness, maximize=True, max_val=2)
    # problem = mlrose.generators.FlipFlopGenerator().generate(seed=1234, size=100)
    # problem = mlrose.generators.TSPGenerator().generate(seed=1234, number_of_cities=10)

    # problem = mlrose.generators.KnapsackGenerator().generate(
    #     seed=1234, number_of_items_types=10)
    # problem = FourPeaksGenerator().generate(seed=1234, size=10)
    # problem = mlrose.generators.MaxKColorGenerator().generate(seed=1234,
    #                                                           number_of_nodes=20)

    methods = [RandomOptimizationMethods.SIMULATED_ANNEALING,
               RandomOptimizationMethods.RANDOMIZED_HILL_CLIMBING,
               RandomOptimizationMethods.GENETIC_ALGORITHM]
    colors = ['r', 'g', 'b']
    handles = []
    for i, method in enumerate(methods):
        _, best_fitness, avg_fitness, avg_iteration, curves = solve(
            problem, method, n_seeds=n_seeds)
        print(
            f"{method.name} - {best_fitness}, {avg_fitness}, {avg_iteration}")
        temp_handles = []
        for curve in curves:
            line, = plt.plot(curve, color=colors[i], label=method.name)
            temp_handles.append(line)
        handles.append(tuple(temp_handles))

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('Optimization Curves')
    plt.legend(handles, [method.name for method in methods])
    plt.savefig(f"optimization_curves.png")
    plt.close()


if __name__ == '__main__':
    main()
