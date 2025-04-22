"""
run_multiple_times.py

Nástroj pro spouštění a vyhodnocování genetického programování (GP) na problému VRPTW
a pro analýzu výsledků. Obsahuje:
  - compute_statistics_for_convergence_files  – výpočet základní statistiky konvergenčních CSV
  - get_csv_filenames                        – generování parametrických názvů CSV souborů
  - run_experiment                           – jeden běh GP pro danou instanci
  - run_multiple_experiments_seq_with_named_csv
                                             – sekvenční série experimentů s pojmenovanými CSV
  - run_multiple_experiments_parallel_with_named_csv
                                             – paralelní série experimentů s pojmenovanými CSV
  - režimy "plot", "run" a "runParallel" v hlavním bloku

Autor:      Petr Kaška
Vytvořeno:  2025-04-22
"""
import matplotlib.pyplot as plt
import random
import sys
import numpy as np
import csv
import os
import multiprocessing
import glob
import re
import pandas as pd
import time

from plot import plot_boxplot_from_all_convergences, plot_all_convergence, plot_all_pareto, plot_average_convergence_from_directory, plot_algorithms_comparison, friedman_on_final_fitness, boxplot_final_distances
from parse_solomon import parse_solomon_xml
from VRPTW import run_gp_for_vrptw


def compute_statistics_for_convergence_files(directory="."):
    """
    Pro všechny soubory 'convergence_*.csv' v zadaném adresáři:
      - Načte data ze sloupce "best_fitness_in_generation"
      - Spočítá průměr, medián, 1. kvartil, 3. kvartil a rozptyl
    Výsledek vytiskne jako tabulku.
    """
    pattern = os.path.join(directory, "convergence_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print("Nenalezeny žádné soubory 'convergence_*.csv'!")
        return

    results = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Chyba při načítání {csv_file}: {e}")
            continue
        
        if "best_fitness_in_generation" not in df.columns:
            print(f"Soubor {csv_file} neobsahuje sloupec 'best_fitness_in_generation'.")
            continue
        
        mean_val = df["best_fitness_in_generation"].mean()
        median_val = df["best_fitness_in_generation"].median()
        q1 = df["best_fitness_in_generation"].quantile(0.25)
        q3 = df["best_fitness_in_generation"].quantile(0.75)
        var_val = df["best_fitness_in_generation"].var()
        max = df["best_fitness_in_generation"].max()
        min = df["best_fitness_in_generation"].min()
        
        file_name = os.path.basename(csv_file)
        results.append({
            "soubor": file_name,
            "průměr": mean_val,
            "medián": median_val,
            "1. kvartil": q1,
            "3. kvartil": q3,
            "rozptyl": var_val,
            "max":max,
            "min":min
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

###############################################################################
# Funkce pro parametrové názvy CSV
###############################################################################
def get_csv_filenames(pop_size, n_gen, cx, mut, shrink, cool, tourn, elite_size, tree_depth_min, tree_depth_max):
    """
    Z parametrů poskládá unikátní jméno pro convergence + final results CSV.
    Např.:
      convergence_pop10_gen20_cx0.9_mut0.3_sh0.2_cf0.995_ts5.csv
      final_results_pop10_gen20_cx0.9_mut0.3_sh0.2_cf0.995_ts5.csv
    """
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    param_str = (
        f"pop{pop_size}_gen{n_gen}_cx{cx}_mut{mut}_"
        f"sh{shrink}_cf{cool}_ts{tourn}_elite{elite_size}_tMin_{tree_depth_min}_tMax_{tree_depth_max}"
    )
    convergence_file = os.path.join(results_dir, f"convergence_{param_str}.csv")
    final_results_file = os.path.join(results_dir, f"final_results_{param_str}.csv")
    return convergence_file, final_results_file

###############################################################################
# Spuštění jedné instance evoluce
###############################################################################
def run_experiment(instance, pop_size, n_gen, run_id,
                   initial_crossover_rate, initial_mutation_rate,
                   shrink_prob, cooling_factor, tournsize, elite_size, tree_depth_min, tree_depth_max, lang):
    """
    Spustí jednu instanci evoluce GP a vrátí slovník s výsledky.
    """
    print(f"\n=== Běh {run_id} ===")
    random.seed()  
    gens, _, best_fits, best_ind, _, total_dist, num_routes = run_gp_for_vrptw(
        instance,
        pop_size=pop_size,
        n_gen=n_gen,
        initial_crossover_rate=initial_crossover_rate,
        initial_mutation_rate=initial_mutation_rate,
        shrink_prob=shrink_prob,
        cooling_factor=cooling_factor,
        tournsize=tournsize,
        elite_size=elite_size,
        tree_depth_min=tree_depth_min,
        tree_depth_max=tree_depth_max,
        lang=lang
    )
    return {
        "run": run_id,
        "gens": gens,
        "best_fits": best_fits,
        "final_fitness": best_ind.fitness.values[0],
        "total_distance": total_dist,
        "number_of_routes": num_routes,
        "best_expression": str(best_ind)
    }

###############################################################################
# Sekvenční varianty s parametrovými CSV (Pro spouštění na mém chábém PC :D)
###############################################################################
def run_multiple_experiments_seq_with_named_csv(instance, num_runs,
                                                pop_size, n_gen,
                                                initial_crossover_rate,
                                                initial_mutation_rate,
                                                shrink_prob,
                                                cooling_factor,
                                                tournsize,
                                                elite_size,
                                                tree_depth_min,
                                                tree_depth_max,
                                                lang):
    """
    Stejné jako run_multiple_experiments_seq, ale výsledky se ukládají
    do CSV pojmenovaných podle parametrů.
    """
    convergence_file, final_file = get_csv_filenames(
        pop_size, n_gen,
        initial_crossover_rate,
        initial_mutation_rate,
        shrink_prob,
        cooling_factor,
        tournsize,
        elite_size,
        tree_depth_min,
        tree_depth_max
    )

    if os.path.exists(convergence_file):
        os.remove(convergence_file)
    if os.path.exists(final_file):
        os.remove(final_file)

    with open(convergence_file, mode='w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(["run", "generation", "best_fitness_in_generation"])
    with open(final_file, mode='w', newline='') as ff:
        writer = csv.writer(ff)
        writer.writerow(["run", "final_fitness", "total_distance",
                         "number_of_routes", "best_expression"])

    experiments = []
    for r in range(1, num_runs + 1):
        exp = run_experiment(
            instance,
            pop_size,
            n_gen,
            r,
            initial_crossover_rate,
            initial_mutation_rate,
            shrink_prob,
            cooling_factor,
            tournsize,
            elite_size,
            tree_depth_min,
            tree_depth_max,
            lang
        )
        experiments.append(exp)

    with open(convergence_file, mode='a', newline='') as cf:
        writer = csv.writer(cf)
        for exp in experiments:
            run_id = exp["run"]
            for g, bf in zip(exp["gens"], exp["best_fits"]):
                writer.writerow([run_id, g, bf])

    with open(final_file, mode='a', newline='') as ff:
        writer = csv.writer(ff)
        for exp in experiments:
            writer.writerow([
                exp["run"],
                exp["final_fitness"],
                exp["total_distance"],
                exp["number_of_routes"],
                exp["best_expression"]
            ])

    print(f"Experimenty dokončeny (SEQ) -> {convergence_file} , {final_file}")

###############################################################################
# Paralelní Spuštění (pro server)
###############################################################################
def run_multiple_experiments_parallel_with_named_csv(instance, num_runs,
                                                     pop_size, n_gen,
                                                     initial_crossover_rate,
                                                     initial_mutation_rate,
                                                     shrink_prob,
                                                     cooling_factor,
                                                     tournsize,
                                                     lang,
                                                     elite_size,
                                                     tree_depth_min,
                                                     tree_depth_max,
                                                     num_workers=None):
    """
    Stejné jako run_multiple_experiments_parallel, ale výsledky se ukládají
    do CSV pojmenovaných podle parametrů.
    """
    convergence_file, final_file = get_csv_filenames(
        pop_size, n_gen,
        initial_crossover_rate,
        initial_mutation_rate,
        shrink_prob,
        cooling_factor,
        tournsize,
        elite_size,
        tree_depth_min,
        tree_depth_max
    )

    if os.path.exists(convergence_file):
        os.remove(convergence_file)
    if os.path.exists(final_file):
        os.remove(final_file)

    with open(convergence_file, mode='w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(["run", "generation", "best_fitness_in_generation"])
    with open(final_file, mode='w', newline='') as ff:
        writer = csv.writer(ff)
        writer.writerow(["run", "final_fitness", "total_distance",
                         "number_of_routes", "best_expression"])

    if num_workers is None:
        num_workers = min(num_runs, os.cpu_count())

    pool = multiprocessing.Pool(processes=num_workers)
    results = []
    for r in range(1, num_runs + 1):
        results.append(pool.apply_async(run_experiment,
                    args=(instance, pop_size, n_gen, r,
                          initial_crossover_rate, initial_mutation_rate,
                          shrink_prob, cooling_factor, tournsize,elite_size, tree_depth_min, tree_depth_max, lang)))
    pool.close()
    pool.join()

    experiments = [res.get() for res in results]

    with open(convergence_file, mode='a', newline='') as cf:
        writer = csv.writer(cf)
        for exp in experiments:
            run_id = exp["run"]
            for g, bf in zip(exp["gens"], exp["best_fits"]):
                writer.writerow([run_id, g, bf])

    with open(final_file, mode='a', newline='') as ff:
        writer = csv.writer(ff)
        for exp in experiments:
            writer.writerow([
                exp["run"],
                exp["final_fitness"],
                exp["total_distance"],
                exp["number_of_routes"],
                exp["best_expression"]
            ])

    print(f"Experimenty dokončeny (PARALLEL) -> {convergence_file} , {final_file}")

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "run"

    num_runs = 4
    pop_size = 100
    n_gen = 4

    best_crossover_rate_so_far = 0.3
    bes_mutation_rate_so_far = 0.8
    best_tournsize = 12
    best_elite_size = 2
    shrink_prob_best = 0.2
    tree_depth_min = 1
    tree_depth_max = 3
    cooling_factor = 0.995

    crossover_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Done
    tournsize_rates = [6, 8, 10, 12] # Done
    elite_size_rates = [1, 2, 3, 4, 10, 15] # Done
    tree_size_min_rates = [1, 2, 3, 4, 5, 6, 7] # Done
    tree_size_max_rates = [3, 4, 5, 6, 7, 8, 9, 10] # Done
    mutation_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Done
    shrink_prob_rates = [0.0, 0.2, 1.0] # Done

    parametry = { # Done
        'crossover_rate': [0.25,0.255,0.26, 0.27,0.28, 0.29, 0.30, 0.31, 0.33, 0.35],
        'mutation_rate': [0.80,0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.88, 0.90],
        'shrink_prob': [0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.25]
    }

    xml_file = "../data/R201_100.xml"
    instance_data = parse_solomon_xml(xml_file)

    path_EVO = "results_extension_to_EVO/"
    # path = "VYSLEDKYYY/fitness_vzdalenost_+_vyska_stromu/convergence_pop1000_gen400_cx0.3_mut0.8_shXX_cf0.995_ts12_elite2_tMin_1_tMax_3"
    path = "VYSLEDKYYY/fitness_vzdalenost_+_vyska_stromu/convergence_pop1000_gen400_cx0.3_mut0.8_sh0.2_cf0.995_ts12_elite2_tMin_XX_tMax_XX"

    if mode == "plot":

        # plot_all_pareto(directory=path)  # nebo "." podle toho, kde CSV leží
        # heur = [
        #     {'label':'Saving heuristika','fitness':1454.91393062,'vehicles':19},
        #     {'label':'Early-First heuristika','fitness':2994.2739324819822,'vehicles':7}
        # ]
        # plot_all_from_directory(directory=path, heuristics=heur)  # nebo "." podle toho, kde CSV leží
        # friedman_on_final_fitness(directory=path_EVO) 
        # plot_algorithms_comparison(data_dir=".") 
        # plot_average_convergence_from_directory(directory=path_EVO) 
        # boxplot_final_distances(directory=path_EVO) 
        # plot_all_convergence(directory=path)  # nebo "." podle toho, kde CSV leží
        # plot_boxplot_from_all_convergences(directory=path)  # Boxplot pro "convergence_*.csv"

        compute_statistics_for_convergence_files(directory=path)


    elif mode == "run":

        print("\n========================")
        print(f"SPUŠTĚNÍ s mutation_rate={tree_depth_max}")
        print("========================")

        run_multiple_experiments_seq_with_named_csv(
            instance_data,
            num_runs=num_runs,
            pop_size=pop_size,
            n_gen=n_gen,
            initial_crossover_rate=best_crossover_rate_so_far,
            initial_mutation_rate=bes_mutation_rate_so_far,
            shrink_prob=shrink_prob_best,
            cooling_factor=cooling_factor,
            tournsize=best_tournsize,
            elite_size=best_elite_size,
            tree_depth_min=tree_depth_min,
            tree_depth_max=tree_depth_max,
            lang=False
        )
        print("Sekvenční série experimentů dokončena.")

    elif mode == "runParallel":

        print("\n========================")
        print(f"SPUŠTĚNÍ PARALELNĚ s tournament_size={num_runs}")
        print("========================")

        run_multiple_experiments_parallel_with_named_csv(
            instance_data,
            num_runs=num_runs,
            pop_size=pop_size,
            n_gen=n_gen,
            initial_crossover_rate=best_crossover_rate_so_far,
            initial_mutation_rate=bes_mutation_rate_so_far,
            shrink_prob=shrink_prob_best,
            cooling_factor=cooling_factor,
            tournsize=best_tournsize,
            elite_size=best_elite_size,
            tree_depth_min=tree_depth_min,
            tree_depth_max=tree_depth_max,
            lang=False
        )

        print("Paralelní série experimentů dokončena.")
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        print(f"Celkový čas běhu: {elapsed_seconds:.2f} s")

    else:
        print("Neznámý režim. Použijte jeden z následujících přepínačů:")
        print("  plot         - pouze vykreslí grafy")
        print("  run          - spustí experimenty sekvenčně")
        print("  runParallel  - spustí experimenty paralelně")