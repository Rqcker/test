import numpy as np
import BenchmarksHub as bh
import pandas as pd
import time
import os


def negative_selection(benchmark_number, pop_size):
    start_time = time.time()
    globalMin, Lb, Ub, nd, max_limit = bh.terminate(benchmark_number)
    detectors = [np.random.uniform(Lb, Ub, nd) for _ in range(pop_size)]

    iterations_number = 0
    best_detector = None
    best_fitness = float('inf')

    while iterations_number < max_limit and best_fitness > globalMin:
        for detector in detectors:
            fitness = bh.benchmark_result(detector, benchmark_number)
            if fitness < best_fitness:
                best_fitness = fitness
                best_detector = detector

        # new detectors
        detectors = [np.random.uniform(Lb, Ub, nd) for _ in range(pop_size)]
        print(f"Iteration {iterations_number}, Best Fitness: {best_fitness}")
        iterations_number += 1

    end_time = time.time()
    running_time = end_time - start_time
    return best_detector, best_fitness, running_time

def run_algorithm_multiple_times(pop_size, benchmark_number, runs):
    results = []

    for run in range(1, runs + 1):
        print(f"Run {run} of {runs} on Benchmark {benchmark_number}")
        best_detector, best_fitness, running_time = negative_selection(benchmark_number, pop_size)
        results.append({'Run': run, 'BestAntibody': best_detector, 'BestFitness': best_fitness})

    return results

def main():
    questions = list(range(1, 27))  # 26 benchmarks
    runs = 1
    pop_size = 50
    results_folder = "nsa_res"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for benchmark_number in questions:
        results = run_algorithm_multiple_times(pop_size, benchmark_number, runs)

        df = pd.DataFrame(results)
        csv_filename = os.path.join(results_folder, f'nsa_{benchmark_number}.csv')
        df.to_csv(csv_filename, index=False)
        print(f'Results for benchmark {benchmark_number} saved in {csv_filename}')

if __name__ == "__main__":
    main()
