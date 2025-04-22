"""
VRPTW.py

Jádro genetického programování (GP) pro VRPTW:
  - distance                             – výpočet euklidovské vzdálenosti
  - construct_solution_with_gp           – greedy výběr zákazníků řízený GP výrazem + Gauss výběr
  - fitness                              – fitness = vzdálenost + penalizace za velikost stromu
  - fitness_langrange                    – Lagrange přístup s penalizací za neobsloužené zákazníky
  - apply_crossover, apply_mutation      – GP operátory křížení a mutace (včetně shrink)
  - apply_tournament_selection           – turnajová selekce s elitismem
  - run_gp_for_vrptw                     – kompletní evoluční běh GP nad jednou instancí

Definuje i DEAP pset s primitivami z `tree_operators` a základní GP nastavení.

Závislosti:
  math, copy, random, deap (base, creator, tools, gp), operator, tree_operators

Author:      Petr Kaška
Created:     2025-04-22
"""
import math
import copy
import random

from deap import base, creator, tools, gp
from operator import attrgetter

import tree_operators as tree

###############################################################################
# Pomocné funkce VRPTW
###############################################################################
def distance(x1, y1, x2, y2):
    """
    Vrátí euklidovskou vzdálenost mezi dvěma body (x1,y1) a (x2,y2),
    vypočtenou pomocí math.hypot.
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def construct_solution_with_gp(gp_func, instance, sigma=1):
    """
    Greedy přidávání zákazníků řízené GP výrazem, ale výběr zákazníka
    podle Gaussovského rozdělení pořadí ve feasible_list.
    sigma = šířka rozptýlení; čím větší, tím častěji se berou i horší kandidáti.
    """
    customers = copy.deepcopy(instance['customers'])
    depot     = instance['depot']
    cap       = instance['vehicle_capacity']
    speed     = instance['vehicle_speed']

    all_routes    = []
    current_route = []
    current_load  = 0.0
    current_time  = 0.0
    last_x, last_y = depot['x'], depot['y']

    while True:
        feasible_list = []
        for c in customers:
            if current_load + c['demand'] > cap:
                continue
            dist = math.hypot(c['x']-last_x, c['y']-last_y)
            arrival = current_time + dist/speed
            if arrival > c['tw_end']:
                continue

            score = gp_func(dist, c['demand'], c['tw_start'], c['tw_end'],
                             current_load, current_time)
            feasible_list.append((c, score, dist, arrival))

        if not feasible_list:
            if current_route:
                back = math.hypot(last_x-depot['x'], last_y-depot['y'])
                current_route.append(("return_depot", back))
                all_routes.append(current_route)
            if (last_x, last_y) == (depot['x'], depot['y']):
                break

            current_route, current_load, current_time = [], 0.0, 0.0
            last_x, last_y = depot['x'], depot['y']
            continue

        feasible_list.sort(key=lambda x: x[1], reverse=True)

        weights = [math.exp(- (i**2) / (2 * sigma**2))
                   for i in range(len(feasible_list))]

        idx = random.choices(range(len(feasible_list)), weights=weights, k=1)[0]
        cust, score, dist_to_c, arr = feasible_list[idx]

        start_service = max(arr, cust['tw_start'])
        finish_service = start_service + cust['service_time']

        current_route.append((cust['id'], dist_to_c))
        current_load   += cust['demand']
        current_time    = finish_service
        last_x, last_y  = cust['x'], cust['y']

        customers.remove(cust)

    total_distance = sum(d for route in all_routes for (_, d) in route)
    return all_routes, total_distance, len(all_routes)


###############################################################################
# DEAP: GP nastavení
###############################################################################
pset = gp.PrimitiveSet("MAIN", 6)
pset.renameArguments(ARG0="dist")
pset.renameArguments(ARG1="demand")
pset.renameArguments(ARG2="tw_start")
pset.renameArguments(ARG3="tw_end")
pset.renameArguments(ARG4="load")
pset.renameArguments(ARG5="current_time")

pset.addPrimitive(tree.add, 2, name="add")
pset.addPrimitive(tree.sub, 2, name="sub")
pset.addPrimitive(tree.mul, 2, name="mul")
pset.addPrimitive(tree.div, 2, name="div")
pset.addPrimitive(tree.log, 1, name="log")
pset.addPrimitive(tree.sqrt, 1, name="sqrt")
pset.addPrimitive(tree.asin, 1, name="asin")
# pset.addPrimitive(tree.neg, 1, name="neg")
# pset.addPrimitive(tree.max_, 2, name="max")
# pset.addPrimitive(tree.min_, 2, name="min")
# pset.addPrimitive(tree.if_then_else, 3, name="if_then_else")    

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("compile", gp.compile, pset=pset)

###############################################################################
# Fitness funkce - původní i "Lagrange"
###############################################################################
def fitness(individual, instance):
    """
    1) Zkompiluje jedince do python funkce (6 vstupů).
    2) Sestaví VRPTW řešení (construct_solution_with_gp).
    3) Vrátí fitness = total_dist + penalizace_za_velikost_stromu.
    """
    gp_func = toolbox.compile(expr=individual)
    _, total_dist, _ = construct_solution_with_gp(gp_func, instance)

    penalty_coefficient = 0.2
    penalty = penalty_coefficient * len(individual)
    total_dist += penalty
    
    return (total_dist,)

def fitness_langrange(
    individual,
    instance,
    toolbox,
    lambda_unserved=1000.0,
    penalty_coefficient=0.2
):
    """
    Vyhodnocení jedince pro VRPTW pomocí tzv. "Lagrange" přístupu.

    Postup:
      1) Zkompilujeme jedince do funkce GP (6 vstupů).
      2) Sestavíme VRPTW řešení (construct_solution_with_gp).
      3) Spočítáme výsledné hodnoty:
         - celkovou ujetou vzdálenost
         - počet obsloužených (a neobsloužených) zákazníků
         - penalizaci za neobsloužené zákazníky (lambda_unserved * unserved)
         - penalizaci za velikost stromu (penalty_coefficient * délka jedince)
      4) Vrátíme fitnes = součet všech penalizací a ujeté vzdálenosti.

    Parametry:
      :param individual:          GP jedinec (strom) k vyhodnocení
      :param instance:            Data VRPTW (zákazníci, depo, kapacita aj.)
      :param toolbox:             DEAP toolbox (kvůli .compile)
      :param lambda_unserved:     Lagrange multiplikátor za neobsloužené zákazníky
      :param penalty_coefficient: Koeficient penalizace za velikost stromu

    Návratová hodnota:
      - n-tice (fitness,) kde fitness je součtem vzdálenosti a penalizací.
    """

    gp_func = toolbox.compile(expr=individual)

    routes, total_dist, _ = construct_solution_with_gp(gp_func, instance)

    served_customers = sum(
        1 for route in routes for step in route if step[0] != "return_depot"
    )
    total_customers = len(instance['customers'])
    unserved_customers = total_customers - served_customers

    unserved_penalty = lambda_unserved * unserved_customers
    size_penalty = penalty_coefficient * len(individual)

    fitness_value = total_dist + unserved_penalty + size_penalty

    return (fitness_value,)


###############################################################################
# GP operátory
###############################################################################
def apply_crossover(offspring, crossover_rate):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_rate:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    return offspring

def apply_mutation(offspring, mutation_rate, shrink_prob):
    for mutant in offspring:
        if random.random() < mutation_rate:
            if random.random() < shrink_prob:
                toolbox.mutate_shrink(mutant)
            else:
                toolbox.mutate(mutant)
            del mutant.fitness.values
    return offspring

def apply_tournament_selection(population, elite_size=2, tournsize=5):
    """
    Turnajová selekce:
      - Zachová elitních jedinců (elite_size).
      - Zbytek populace se naplní vítězi z turnajů, kde se z náhodně vybrané skupiny 
        (velikosti tournsize) vybere jedinec s nejnižší fitness.
      
    Předpokládá se, že nižší fitness je lepší.
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
    next_gen = [toolbox.clone(ind) for ind in sorted_pop[:elite_size]]
    
    for _ in range(len(population) - elite_size):
        tournament = random.sample(population, tournsize)
        winner = min(tournament, key=lambda ind: ind.fitness.values[0])
        next_gen.append(toolbox.clone(winner))
        
    return next_gen

###############################################################################
# Hlavní funkce pro GP
###############################################################################

def run_gp_for_vrptw(
    instance, 
    pop_size, 
    n_gen, 
    initial_crossover_rate,
    initial_mutation_rate,
    shrink_prob,
    cooling_factor,
    tournsize,
    elite_size,
    tree_depth_min=0,
    tree_depth_max=3,
    lang=False
):
    """
    Spustí evoluční proces GP pro danou instanci VRPTW.

    - lang=False (výchozí) => fitness = vzdálenost + 0.2*délka_stromu
    - lang=True  => fitness = vzdálenost + 1000*(neobsloužení) + 0.2*délka_stromu

    Vrací: 
      (generations, current_fitness_values, best_fitness_values, best_ind, routes, total_dist, number_of_routes)
    """

    if lang:
        toolbox.register("evaluate", lambda individual: fitness_langrange(individual, instance, toolbox))
    else:
        toolbox.register("evaluate", fitness, instance=instance)

    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=tree_depth_min, max_=tree_depth_max)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.3)
    toolbox.register("mutate_shrink", gp.mutShrink)

    pop = toolbox.population(n=pop_size)

    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    generations = []
    best_fitness_values = []
    current_fitness_values = []
    best_ind = None

    crossover_rate = initial_crossover_rate
    mutation_rate = initial_mutation_rate

    for gen in range(1, n_gen + 1):
        offspring = apply_tournament_selection(pop, elite_size, tournsize=tournsize)
        offspring = apply_crossover(offspring, crossover_rate=crossover_rate)
        offspring = apply_mutation(offspring, mutation_rate=mutation_rate, shrink_prob=shrink_prob)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        current_best = tools.selBest(pop, 1)[0]
        current_fitness = current_best.fitness.values[0]
        if best_ind is None or current_fitness < best_ind.fitness.values[0]:
            best_ind = toolbox.clone(current_best)

        generations.append(gen)
        current_fitness_values.append(current_fitness)
        best_fitness_values.append(best_ind.fitness.values[0])

        if (gen % 5) == 0:
            print(f"\rGenerace {gen}/{n_gen} - curr fitness: {current_fitness:.4f} | "f"shrink_prob={shrink_prob:.2f}", end='', flush=True)
            if gen == n_gen:
                print()  

        if gen < n_gen - 100:
            crossover_rate *= cooling_factor 
            mutation_rate *= cooling_factor 

    final_func = toolbox.compile(expr=best_ind)
    routes, total_dist, number_of_routes = construct_solution_with_gp(final_func, instance)

    return generations, current_fitness_values, best_fitness_values, best_ind, routes, total_dist, number_of_routes