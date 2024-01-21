import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm
def bound(a, upper_bound, lower_bound):
    a[a > upper_bound] = upper_bound[a > upper_bound]
    a[a < lower_bound] = lower_bound[a < lower_bound]
    return a

### Benchmarks
# F1 = Beale [-4.5; 4.5]; 0
# F2 = Easom [-100,100]; -1
# F3 = Matyas [-10,10]; 0
# F4 = Bochachvesky 1 [-100,100]; 0
# F5 = Booth [-10, 10]; 0
# F6 = Michalewicz 2[0,pi]; -1.8013
# F7 = Schaffer [-100; 100]; 0
# F8 = Six Hump Camel Back [-5; 5]; -1.03163
# F9 = Bochachvesky 2 [-100,100]; 0
# F10 = Bochachvesky 3 [-100,100]; 0
# F11 = Shubert [-10,10]; -186.73
# F12 = Colville [-10,10]; 0
# F13 = Michalewicz 5 [0,pi]; -4.6877
# F14 = Zakharov[-5,10]; 0
# F15 = Michalewicz 10 [0,pi]; -4.6877
# F16 = Step [-5.12; 5.12]; 0
# F17 = Sphere [-100,100]; 0
# F18 = SumSquares [-10, 10]; 0
# F19 = Quartic [-1.28,1.28]; 0
# F20 = Schwefel 2.22 [-10,10]; 0
# F21 = Schwefel 1.2 [-10,10]; 0
# F22 = Rosenbrock [-30,30]; 0
# F23 = Dixon-Price [-10, 10]; 0
# F24 = Rastrigin [-5.12; 5.12];
# F25 = Griewank [-600,600]; 0
# F26 = Ackley [-600; 600]; 0


def benchmark_result(x, benchmark_number):
    if benchmark_number == 1:
        y = ((1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2)

    elif benchmark_number == 2:
        y = -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))

    elif benchmark_number == 3:
        y = 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

    elif benchmark_number == 4:
        y = x[0]**2 + 2 * x[1]**2 - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7

    elif benchmark_number == 5:
        y = (x[0]**2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

    elif benchmark_number == 6:
      m = 10
      d = len(x)
      sum = 0
      for i in range(d):
        xi = x[i]
        new_term = np.sin(xi) * np.sin((i + 1) * xi**2 / np.pi)**(2 * m)
        sum += new_term
      y = -sum

    elif benchmark_number == 7:
        y = 0.5 + (np.sin((x[0]**2 + x[1]**2)**0.5)**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2)**2)

    elif benchmark_number == 8:
        y = 4 * x[0]**2 - 2.1 * x[0]**4 + 1/3 * x[0]**6 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4

    elif benchmark_number == 9:
        y = x[0]**2 + 2 * x[1]**2 - 0.3 * np.cos((3 * np.pi * x[0]) * (4 * np.pi * x[1])) + 0.3

    elif benchmark_number == 10:
        y = x[0]**2 + 2 * x[1]**2 - 0.3 * np.cos((3 * np.pi * x[0]) + (4 * np.pi * x[1])) + 0.3

    elif benchmark_number == 11:
        sum1 = 0
        sum2 = 0
        for i in range(1, 6):
            sum1 += i * np.cos((i + 1) * x[0] + i)
            sum2 += i * np.cos((i + 1) * x[1] + i)
        y = sum1 * sum2

    elif benchmark_number == 12:
        term1 = 100 * (x[0]**2 - x[1])**2
        term2 = (x[0] - 1)**2
        term3 = (x[2] - 1)**2
        term4 = 90 * (x[2]**2 - x[3])**2
        term5 = 10.1 * ((x[1] - 1)**2 + (x[3] - 1)**2)
        term6 = 19.8 * (x[1] - 1) * (x[3] - 1)
        y = term1 + term2 + term3 + term4 + term5 + term6

    elif benchmark_number == 13:
      m = 10
      d = len(x)
      sum = 0
      for i in range(d):
        xi = x[i]
        new_term = np.sin(xi) * np.sin((i + 1) * xi**2 / np.pi)**(2 * m)
        sum += new_term
      y = -sum

    elif benchmark_number == 14:
        n = len(x)
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i in range(n):
            sum1 += x[i]**2
            sum2 += 0.5 * i * x[i]
            sum3 += 0.5 * i * x[i]
        y = sum1 + (sum2)**2 + (sum3)**4

    elif benchmark_number == 15:
      m = 10
      d = len(x)
      sum = 0
      for i in range(d):
        xi = x[i]
        new_term = np.sin(xi) * np.sin((i + 1) * xi**2 / np.pi)**(2 * m)
        sum += new_term
      y = -sum

    elif benchmark_number == 16:
        n = len(x)
        sum1 = 0
        for i in range(n):
            sum1 += (x[i] + 0.5)**2
        y = sum1

    elif benchmark_number == 17:
        y = np.sum(np.array(x)**2)

    elif benchmark_number == 18:
        n = len(x)
        sum1 = 0
        for i in range(n):
            sum1 += i * x[i]**2
        y = sum1

    elif benchmark_number == 19:
        n = len(x)
        sum1 = 0
        for i in range(n):
            sum1 += i * x[i]**4
        y = sum1 + np.random.rand()

    elif benchmark_number == 20:
        n = len(x)
        sum1 = 0
        sum2 = 1
        for i in range(n):
            sum1 += (x[i]**2)**0.5
            sum2 *= (x[i]**2)**0.5
        y = sum1 + sum2

    elif benchmark_number == 21:
        n = len(x)
        sum1 = 0
        for i in range(n):
            for j in range(1, i + 1):
                sum1 += x[j]**2
        y = sum1

    elif benchmark_number == 22:
        n = len(x)
        y = 0
        for i in range(n - 1):
            y += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2

    elif benchmark_number == 23:
        n = len(x)
        sum1 = (x[0] - 1)**2
        sum2 = 0
        for i in range(1, n):
            sum2 += i * (2 * x[i]**2 - x[i - 1])**2
        y = sum1 + sum2

    elif benchmark_number == 24:
        n = len(x)
        s = 0
        for j in range(n):
            s += (x[j]**2 - 10 * np.cos(2 * np.pi * x[j]))
        y = 10 * n + s

    elif benchmark_number == 25:
        n = len(x)
        fr = 4000
        s = 0
        p = 1
        for j in range(n):
            s += x[j]**2
        for j in range(n):
            p *= np.cos(x[j] / np.sqrt(j + 1))
        y = s / fr - p + 1

    elif benchmark_number == 26:
        n = len(x)
        a = 20
        b = 0.2
        c = 2 * np.pi
        s1 = 0
        s2 = 0
        for i in range(n):
            s1 += x[i]**2
            s2 += np.cos(c * x[i])
        y = -a * np.exp(-b * np.sqrt(1 / n * s1)) - np.exp(1 / n * s2) + a + np.exp(1)

    return y


def terminate(benchmark_number):
    max_limit = 50  # maximum number of function evaluations
    Tol = 1e-12

    if benchmark_number == 1:
        Lb = -4.5
        Ub = 4.5
        nd = 2  # number of variables
        Lb = np.ones(nd) * Lb  # upper bound
        Ub = np.ones(nd) * Ub  # lower bound
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 2:
        Lb = -100
        Ub = 100
        nd = 2
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = -1
        globalMin += Tol
    elif benchmark_number == 3:
        Lb = -10
        Ub = 10
        nd = 2
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 4:
        Lb = -100
        Ub = 100
        nd = 2
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 5:
        Lb = -10
        Ub = 10
        nd = 2
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 6:
        Lb = 0
        Ub = np.pi
        nd = 2
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = -1.8013
        globalMin += Tol
    elif benchmark_number == 7:
        Lb = -100
        Ub = 100
        nd = 2
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 8:
        Lb = -5
        Ub = 5
        nd = 2
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = -1.0316284534898
        globalMin += Tol
    elif benchmark_number == 9:
        Lb = -100
        Ub = 100
        nd = 2
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 10:
        Lb = -100
        Ub = 100
        nd = 2
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 11:
        Lb = -10
        Ub = 10
        nd = 2
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = -186.73
        globalMin += Tol
    elif benchmark_number == 12:
        Lb = -10
        Ub = 10
        nd = 4
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 13:
        Lb = 0
        Ub = np.pi
        nd = 5
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = -4.687658
        globalMin += Tol
    elif benchmark_number == 14:
        Lb = -5
        Ub = 10
        nd = 10
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 15:
        Lb = 0
        Ub = np.pi
        nd = 10
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = -9.660151715641349
        globalMin += Tol
    elif benchmark_number == 16:
        Lb = -5.12
        Ub = 5.12
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 17:
        Lb = -100
        Ub = 100
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 18:
        Lb = -10
        Ub = 10
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 19:
        Lb = -1.28
        Ub = 1.28
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 20:
        Lb = -10
        Ub = 10
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 21:
        Lb = -100
        Ub = 100
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 22:
        Lb = -30
        Ub = 30
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 23:
        Lb = -10
        Ub = 10
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 24:
        Lb = -5.12
        Ub = 5.12
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 25:
        Lb = -600
        Ub = 600
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol
    elif benchmark_number == 26:
        Lb = -32
        Ub = 32
        nd = 30
        Lb = np.ones(nd) * Lb
        Ub = np.ones(nd) * Ub
        globalMin = 0
        globalMin += Tol

    return globalMin, Lb, Ub, nd, max_limit
    

def run_algorithm_multiple_times(population_size, benchmark_number, runs):
    results = []

    for run in range(1, runs + 1):
        print(f"Run {run} of {runs}")
        best_antibody, best_fitness = SAIS(population_size, benchmark_number)
        results.append({'Run': run, 'BestAntibody': best_antibody, 'BestFitness': best_fitness})

    return results

# fixed SAIS to fit our lab
def SAIS(population_size, benchmark_number):
    print('SAIS Started...')
    start_time = time.time()
    # initialize the SAIS algorithm 3 core parts size
    symbiotic_population_size = int(population_size / 3)

    # initialize the population of solutions
    population = []
    global_min, lower_bound, upper_bound, nd, max_limit = terminate(benchmark_number)
    for i in range(population_size):
        x = np.random.uniform(lower_bound, upper_bound, nd)
        y = benchmark_result(x, benchmark_number)
        solution = (x, y)
        population.append(solution)

    # iterate through the Symbiotic AIS optimization process
    iterations_number = 0
    while iterations_number < max_limit:

      # initialize memory_population (clone)
      memory_population = population.copy()

      # initialize Mutualism, Commensalism and Parasitism populations
      mutualism_population = population[:symbiotic_population_size]
      commensalism_population = population[symbiotic_population_size:2*symbiotic_population_size]
      parasitism_population = population[2*symbiotic_population_size:3*symbiotic_population_size]

      # if the size of the population is not divisible by 3,
      # then use the remainder as the part4
      if len(population) % 3 != 0:
        old_part = population[3*symbiotic_population_size:]
      else:
        old_part = []

      # removing 3 parts from the population
      for part in [mutualism_population, commensalism_population, parasitism_population]:
        for item in part:
          if item in population:
            population.remove(item)

      ### mutualism group start

      # update the best antibodies mutualism_population
      sorted_indices_mu = np.argsort([i[1] for i in mutualism_population])
      best_antibodies_mu = [mutualism_population[j][0] for j in sorted_indices_mu[:symbiotic_population_size]]
      best_fitness_list_mu = [mutualism_population[j][1] for j in sorted_indices_mu[:symbiotic_population_size]]

      # select the best one
      # best_fitness is my breakpoint test
      best_fitness_mu, idx_mu = min((v, i) for i, v in enumerate(best_fitness_list_mu))
      best_antibody_mu = best_antibodies_mu[idx_mu]

      # define the benefit factor 1 or 2
      bf = np.random.randint(1, 3, size=nd)

      # update best antibodies
      for j in range(symbiotic_population_size):
        i = np.random.randint(0, symbiotic_population_size)
        if i == j:
          i = np.random.randint(0, symbiotic_population_size)
        # define mutual factor
        mu = (best_antibodies_mu[i] + best_antibodies_mu[j]) / 2
        # get previous fitness for comparison
        old_fitness_i = benchmark_result(best_antibodies_mu[i], benchmark_number)
        old_fitness_j = benchmark_result(best_antibodies_mu[j], benchmark_number)
        # calculate new solution
        new_antibody_i = best_antibodies_mu[i] + np.random.uniform(0, 1, nd) * (best_antibody_mu - mu*bf)
        new_antibody_i = bound(new_antibody_i, upper_bound, lower_bound)
        new_antibody_j = best_antibodies_mu[j] + np.random.uniform(0, 1, nd) * (best_antibody_mu - mu*bf)
        new_antibody_j = bound(new_antibody_j, upper_bound, lower_bound)
        # evaluate the fitness of the new solution
        new_fitness_i = benchmark_result(new_antibody_i, benchmark_number)
        new_fitness_j = benchmark_result(new_antibody_j, benchmark_number)
        # comparison:
        if new_fitness_i < old_fitness_i:
          # replace old antibody and fitness with new ones
          population.append((new_antibody_i, new_fitness_i))
        elif new_fitness_i >= old_fitness_i:
          population.append((best_antibodies_mu[i], old_fitness_i))
        elif new_fitness_j < old_fitness_j:
          # replace old antibody and fitness with new ones
          population.append((new_antibody_j, new_fitness_j))
        elif new_fitness_j >= old_fitness_j:
          population.append((best_antibodies_mu[j], old_fitness_j))

      ### mutualism group end

      ### commensalism group start
      # update the best antibodies commensalism_population
      sorted_indices_co = np.argsort([i[1] for i in commensalism_population])
      best_antibodies_co = [commensalism_population[j][0] for j in sorted_indices_co[:symbiotic_population_size]]
      best_fitness_list_co = [commensalism_population[j][1] for j in sorted_indices_co[:symbiotic_population_size]]

      # select the best one
      # best_fitness is my breakpoint test
      best_fitness_co, idx_co = min((v, i) for i, v in enumerate(best_fitness_list_co))
      best_antibody_co = best_antibodies_co[idx_co]

      # update best antibodies
      for j in range(symbiotic_population_size):
        i = np.random.randint(0, symbiotic_population_size)
        if i == j:
          i = np.random.randint(0, symbiotic_population_size)
        # get previous fitness for comparison
        old_fitness_co = benchmark_result(best_antibodies_co[j], benchmark_number)
        # calculate new solution
        new_antibody_co = best_antibodies_co[j] + np.random.uniform(-1, 1, nd) * (best_antibody_co - best_antibodies_co[i])
        new_antibody_co = bound(new_antibody_co, upper_bound, lower_bound)
        # evaluate the fitness of the new solution
        new_fitness_co = benchmark_result(new_antibody_co, benchmark_number)
        # comparison:
        if new_fitness_co < old_fitness_co:
          # replace old antibody and fitness with new ones
          population.append((new_antibody_co, new_fitness_co))
        else:
          population.append((best_antibodies_co[j], old_fitness_co))

      ### commensalism group end

      ### parasitism group start
      # update the best antibodies parasitism_population
      sorted_indices_pa = np.argsort([i[1] for i in parasitism_population])
      best_antibodies_pa = [parasitism_population[j][0] for j in sorted_indices_pa[:symbiotic_population_size]]
      best_fitness_list_pa = [parasitism_population[j][1] for j in sorted_indices_pa[:symbiotic_population_size]]

      # select the best one
      # best_fitness is my breakpoint test
      best_fitness_pa, idx_pa = min((v, i) for i, v in enumerate(best_fitness_list_pa))
      best_antibody_pa = best_antibodies_pa[idx_pa]

      # update best antibodies
      for j in range(symbiotic_population_size):
        i = np.random.randint(0, symbiotic_population_size)
        if i == j:
          i = np.random.randint(0, symbiotic_population_size)
        # get p and q fitness for comparison
        p_fitness = benchmark_result(best_antibodies_pa[j], benchmark_number)
        q_fitness = benchmark_result(best_antibodies_pa[i], benchmark_number)
        # calculate new solution
        if p_fitness < q_fitness:
          best_antibodies_pa[i] = best_antibodies_pa[j]
          new_antibody_pa = best_antibodies_pa[j]
          new_fitness_pa = benchmark_result(new_antibody_pa, benchmark_number)
        else:
          best_antibodies_pa[j] = best_antibodies_pa[i]
          new_antibody_pa = best_antibodies_pa[i]
          new_fitness_pa = benchmark_result(new_antibody_pa, benchmark_number)

        # replace old antibody and fitness with new ones
        population.append((new_antibody_pa, new_fitness_pa))

      ### parasitism group end

      population = np.vstack((population, memory_population))
      # update the best antibodies population quick way :)
      best_solution = min(population, key=lambda x: x[1])
      best_fitness = best_solution[1]
      # sorted popualtion which with double size
      population = sorted(population, key=lambda x: x[1])
      half_size = len(population) // 2
      # make sure same size as old population and get top antibodies
      population = population[:half_size]


      # check if termination conditions are met
      if best_fitness < global_min:
        print('Iterations Number:', iterations_number)
        break

      # increase the number of function evaluation counter
      iterations_number += 1

    # update the best antibody
    best_fitness, idx = min((s[1], i) for i, s in enumerate(population))
    best_antibody = population[idx][0]

    # update the end time
    end_time = time.time()
    # result display
    print('Running Time: %s Secounds' % (end_time - start_time))
    print("Best Fitness:", best_fitness)
    print("Best Antibody:", best_antibody)

    # get the best solution
    return best_antibody, best_fitness

# define the running number
questions = [1,2,3,4,5,6,7,8,9,10,11,14,16,17,18,20,21,24,25,26,13,12,15,19,22,23]
runs = 1
population_size = 500000
for benchmark_number in questions:
    results = run_algorithm_multiple_times(population_size, benchmark_number, runs)

    # write the results to csv
    df = pd.DataFrame(results)
    csv_filename = f'sais_pop{population_size}_{benchmark_number}.csv'
    df.to_csv(csv_filename, index=False)
    print(f'Results for benchmark {benchmark_number} saved in {csv_filename}')