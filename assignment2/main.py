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
from time import time


class OptimizationProblems(Enum):
    ONE_MAX = 1
    MAX_K_COLOR = 2
    FOUR_PEAKS = 3
    FLIP_FLOP = 4
    TSP = 5
    KNAPSACK = 6
    QUEENS = 7


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
                problem=problem, schedule=mlrose.ExpDecay(init_temp=10), random_state=seed, max_iters=max_iters, curve=True)
        elif method == RandomOptimizationMethods.RANDOMIZED_HILL_CLIMBING:
            state, fitness, curve = mlrose.random_hill_climb(
                problem=problem, random_state=seed, restarts=5, max_iters=max_iters, curve=True)
        elif method == RandomOptimizationMethods.GENETIC_ALGORITHM:
            state, fitness, curve = mlrose.genetic_alg(
                problem=problem, pop_size=200, mutation_prob=0.1, max_iters=max_iters, random_state=seed, curve=True)
        elif method == RandomOptimizationMethods.MIMIC:
            state, fitness, curve = mlrose.mimic(
                problem=problem, pop_size=200, max_iters=max_iters, keep_pct=0.2, random_state=seed, curve=True)
        else:
            raise NotImplementedError()
        if best_fitness is None or fitness > best_fitness:
            best_fitness = fitness
            best_state = state
        fitness_results.append(fitness)
        iterations.append(curve.shape[0])
        curves.append(curve[:, 0])
    return best_state, best_fitness, np.mean(fitness_results), np.mean(iterations), curves


def plot_optimize_curve(opt_problem=OptimizationProblems.KNAPSACK, n_seeds=10):
    exec_times = {RandomOptimizationMethods.SIMULATED_ANNEALING: [
    ], RandomOptimizationMethods.RANDOMIZED_HILL_CLIMBING: [], RandomOptimizationMethods.GENETIC_ALGORITHM: []}
    avg_fitnesses = {RandomOptimizationMethods.SIMULATED_ANNEALING: [
    ], RandomOptimizationMethods.RANDOMIZED_HILL_CLIMBING: [], RandomOptimizationMethods.GENETIC_ALGORITHM: []}

    N = np.logspace(1, 5, 5, dtype=int)
    for n_items in N:
        if opt_problem == OptimizationProblems.ONE_MAX:
            fitness = mlrose.OneMax()
            problem = mlrose.DiscreteOpt(
                length=n_items, fitness_fn=fitness, maximize=True, max_val=2)
        elif opt_problem == OptimizationProblems.MAX_K_COLOR:
            problem = mlrose.generators.MaxKColorGenerator().generate(seed=1234,
                                                                      number_of_nodes=n_items, maximize=True)
        elif opt_problem == OptimizationProblems.FOUR_PEAKS:
            problem = FourPeaksGenerator().generate(seed=1234, size=n_items)
        elif opt_problem == OptimizationProblems.FLIP_FLOP:
            problem = mlrose.generators.FlipFlopGenerator().generate(seed=1234, size=n_items)
        elif opt_problem == OptimizationProblems.TSP:
            problem = mlrose.generators.TSPGenerator().generate(
                seed=1234, number_of_cities=n_items)
        elif opt_problem == OptimizationProblems.KNAPSACK:
            problem = mlrose.generators.KnapsackGenerator().generate(
                seed=1234, number_of_items_types=n_items)
        elif opt_problem == OptimizationProblems.QUEENS:
            problem = mlrose.generators.QueensGenerator().generate(
                seed=1234, size=n_items, maximize=True)

        methods = [RandomOptimizationMethods.SIMULATED_ANNEALING,
                   RandomOptimizationMethods.RANDOMIZED_HILL_CLIMBING, RandomOptimizationMethods.GENETIC_ALGORITHM]

        for _, method in enumerate(methods):
            start_time = time()
            _, best_fitness, avg_fitness, avg_iteration, _ = solve(
                problem, method, n_seeds=n_seeds, max_iters=100)
            exec_time = (time() - start_time) / n_seeds
            print(
                f"{method.name} - {best_fitness}, {avg_fitness}, {avg_iteration}, {exec_time}")
            exec_times[method].append(exec_time)
            avg_fitnesses[method].append(avg_fitness)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    labels = {RandomOptimizationMethods.SIMULATED_ANNEALING: 'SA',
              RandomOptimizationMethods.RANDOMIZED_HILL_CLIMBING: 'RHC',
              RandomOptimizationMethods.GENETIC_ALGORITHM: 'GA'}
    plt.subplot(1, 2, 1)
    for item in exec_times.items():
        key, value = item
        plt.plot(N, value, label=labels[key])
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Problem size')
    plt.ylabel('Execution time (sec)')
    plt.title("Execute Time")
    plt.subplot(1, 2, 2)
    for item in avg_fitnesses.items():
        key, value = item
        plt.plot(N, value, label=labels[key])
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Problem size')
    plt.ylabel('Fitness')
    plt.title("Fitness")
    plt.savefig(os.path.join('checkpoints',
                f"perf_{opt_problem.name}.png"))
    plt.close()


def main():
    '''
    '''
    opt_problems = [OptimizationProblems.KNAPSACK]
    n_seeds = 10

    for opt_problem in opt_problems:
        if opt_problem == OptimizationProblems.ONE_MAX:
            fitness = mlrose.OneMax()
            problem = mlrose.DiscreteOpt(
                length=10000, fitness_fn=fitness, maximize=True, max_val=2)
        elif opt_problem == OptimizationProblems.MAX_K_COLOR:
            problem = mlrose.generators.MaxKColorGenerator().generate(seed=1234,
                                                                      number_of_nodes=100, maximize=True)
        elif opt_problem == OptimizationProblems.FOUR_PEAKS:
            problem = FourPeaksGenerator().generate(seed=1234, size=100)
        elif opt_problem == OptimizationProblems.FLIP_FLOP:
            problem = mlrose.generators.FlipFlopGenerator().generate(seed=1234, size=100)
        elif opt_problem == OptimizationProblems.TSP:
            problem = mlrose.generators.TSPGenerator().generate(seed=1234, number_of_cities=10)
        elif opt_problem == OptimizationProblems.KNAPSACK:
            problem = mlrose.generators.KnapsackGenerator().generate(
                seed=1234, number_of_items_types=100)
        elif opt_problem == OptimizationProblems.QUEENS:
            problem = mlrose.generators.QueensGenerator().generate(
                seed=1234, size=20, maximize=True)

        methods = [RandomOptimizationMethods.SIMULATED_ANNEALING,
                   RandomOptimizationMethods.RANDOMIZED_HILL_CLIMBING,
                   RandomOptimizationMethods.GENETIC_ALGORITHM]
        colors = ['r', 'g', 'b']
        handles = []
        for i, method in enumerate(methods):
            start_time = time()
            _, best_fitness, avg_fitness, avg_iteration, curves = solve(
                problem, method, n_seeds=n_seeds)
            end_time = time()
            print(
                f"{method.name} - {best_fitness}, {avg_fitness}, {avg_iteration}, {(end_time - start_time) / n_seeds}")
            temp_handles = []
            for curve in curves:
                line, = plt.plot(curve, color=colors[i], label=method.name)
                temp_handles.append(line)
            handles.append(tuple(temp_handles))

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.title(f"Optimization Curves\n{opt_problem.name}")
        plt.legend(handles, [method.name for method in methods])
        plt.savefig(os.path.join('checkpoints',
                    f"optimization_curves_{opt_problem.name}.png"))
        plt.close()


if __name__ == '__main__':
    plot_optimize_curve(OptimizationProblems.KNAPSACK)
    plot_optimize_curve(OptimizationProblems.ONE_MAX)
