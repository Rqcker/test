import numpy as np
import BenchmarksHub as bh
import pandas as pd
import time
import os


#test!!!
print('NONONONO')
print('NONONONO')
#print('yes, it works!')
def clonal_selection(benchmark_number, pop_size, clone_factor, mutation_rate):
    start_time = time.time()
    globalMin, Lb, Ub, nd, max_limit = bh.terminate(benchmark_number)
    population = [np.random.uniform(Lb, Ub, nd) for _ in range(pop_size)]

    iterations_number = 0
    best_solution = None
    best_fitness = float('inf')

    while iterations_number < max_limit and best_fitness > globalMin:
        fitness = [bh.benchmark_result(ind, benchmark_number) for ind in population]
        ranked_population = sorted(zip(population, fitness), key=lambda x: x[1])
        best_solution, best_fitness = ranked_population[0]

        clones = []
        for i, (vector, _) in enumerate(ranked_population):
            n_clones = int(np.round(clone_factor / (i + 1)))
            clones.extend([vector.copy() for _ in range(n_clones)])

        for clone in clones:
            if np.random.rand() < mutation_rate:
                clone += np.random.normal(0, 1, nd)
                clone = bh.bound(clone, Ub, Lb)

        clone_fitness = [bh.benchmark_result(clone, benchmark_number) for clone in clones]
        population = [x for x, _ in sorted(zip(clones, clone_fitness), key=lambda x: x[1])[:pop_size]]

        iterations_number += 1

    end_time = time.time()
    running_time = end_time - start_time
    return best_solution, best_fitness, running_time

def run_algorithm_multiple_times(pop_size, clone_factor, mutation_rate, benchmark_number, runs):
    results = []

    for run in range(1, runs + 1):
        print(f"Run {run} of {runs} on Benchmark {benchmark_number}")
        best_antibody, best_fitness, running_time = clonal_selection(benchmark_number, pop_size, clone_factor, mutation_rate)
        results.append({'Run': run, 'BestAntibody': best_antibody, 'BestFitness': best_fitness})

    return results

def main():
    questions = list(range(1, 27))  # 26 benchmarks
    runs = 1
    pop_size = 50
    clone_factor = 10
    mutation_rate = 0.1
    results_folder = "clonal_res"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for benchmark_number in questions:
        results = run_algorithm_multiple_times(pop_size, clone_factor, mutation_rate, benchmark_number, runs)

        df = pd.DataFrame(results)
        csv_filename = os.path.join(results_folder, f'clonalg_{benchmark_number}.csv')
        df.to_csv(csv_filename, index=False)
        print(f'Results for benchmark {benchmark_number} saved in {csv_filename}')

if __name__ == "__main__":
    main()
