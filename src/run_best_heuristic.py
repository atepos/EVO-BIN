"""
run_best_heuristic.py

Nástroj pro výběr nejlepšího GP stromu z CSV a jeho aplikaci na VRPTW instance:
  - compute_depth              – spočítá maximální hloubku stromového výrazu
  - save_best_tree_from_csv    – vybere z CSV strom s nejnižším fitness (a nejmenší hloubkou) a uloží ho do pickle
  - batch_test_best_tree       – načte uložený strom, zkompiluje ho a otestuje na všech instancích v adresáři
  - main                       – řídí tok: vytvoření pickle, dávkový nebo jednorázový test

Author:      Petr Kaška
Created:     2025-04-22
"""
import sys
import os
import glob
import csv
import pickle

from parse_solomon import parse_solomon_xml
from VRPTW import toolbox, construct_solution_with_gp

def compute_depth(expr):
    """
    Spočte maximální hloubku stromu (počet vnořených závorek) v zadaném výrazu.
    Každá otevírací závorka '(' zvyšuje aktuální hloubku o 1,
    každá zavírací ')' ji snižuje o 1.
    Vrací nejvyšší dosaženou hodnotu hloubky.
    """
    depth = max_depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    return max_depth

def save_best_tree_from_csv(csv_file, pkl_file):
    """
    Ze zadaného CSV souboru (s poli 'final_fitness' a 'best_expression')
    vybere strom s nejnižším fitness; v případě shody i s nejmenší hloubkou.
    Výsledek (string výrazu) uloží pomocí pickle do souboru pkl_file.
    """
    best_fitness = float('inf')
    best_depth   = float('inf')
    best_expr    = None

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fit = float(row["final_fitness"])
            except:
                continue
            expr = row["best_expression"]
            d = compute_depth(expr)
            if fit < best_fitness or (fit == best_fitness and d < best_depth):
                best_fitness, best_depth, best_expr = fit, d, expr

    if best_expr is None:
        raise RuntimeError("Nenalezen žádný validní strom v CSV")

    with open(pkl_file, "wb") as pf:
        pickle.dump(best_expr, pf)
    print(f"Uloženo do {pkl_file}: fitness={best_fitness:.2f}, depth={best_depth}")

def batch_test_best_tree(inst_dir, best_tree_file, out_csv):
    """
    Pro každý XML soubor instancí v adresáři inst_dir:
      1) načte nejlepší strom z best_tree_file (pickle)
      2) zkompiluje GP-funkci
      3) aplikuje ji na instanci přes construct_solution_with_gp
      4) zapíše výsledky (velikost, vzdálenost, počet tras) do out_csv
    """
    with open(best_tree_file, "rb") as f:
        best_tree = pickle.load(f)
    best_func = toolbox.compile(expr=best_tree)

    xmls = sorted(glob.glob(os.path.join(inst_dir, "R2*_*.xml")))
    if not xmls:
        print("Žádné instanční XML ve složce", inst_dir)
        return

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "total_distance", "num_routes"])
        for xml in xmls:
            size = os.path.basename(xml).split("_")[1].replace(".xml","")
            inst = parse_solomon_xml(xml)
            routes, total_dist, num_routes = construct_solution_with_gp(best_func, inst)
            writer.writerow([size, f"{total_dist:.2f}", num_routes])
            print(f"{xml:30s} size={size} dist={total_dist:.2f} routes={num_routes}")

def main():
    """
    Hlavní funkce skriptu:
      - Pokud pickle se stromem neexistuje, zavolá save_best_tree_from_csv
      - Pokud jsou předány 2 argumenty (inst_dir a out_csv), spustí batch_test_best_tree
      - Jinak provede jednorázový test na jedné pevné instanci
    """
    final_csv     = "here.csv"
    best_tree_pkl = "best_tree.pkl"

    if not os.path.exists(best_tree_pkl):
        save_best_tree_from_csv(final_csv, best_tree_pkl)

    if len(sys.argv) == 3:
        inst_dir = sys.argv[1]
        out_csv  = sys.argv[2]
        batch_test_best_tree(inst_dir, best_tree_pkl, out_csv)
    else:
        xml = "../data/R201_025.xml"
        inst = parse_solomon_xml(xml)
        with open(best_tree_pkl, "rb") as f:
            best_tree = pickle.load(f)
        best_func = toolbox.compile(expr=best_tree)
        routes, total_dist, num_routes = construct_solution_with_gp(best_func, inst)

        print(f"Test na {xml}:")
        print("Trasy:", len(routes))
        for i, r in enumerate(routes,1):
            print(" ",r)
        print(f"Celkem km: {total_dist:.2f}")

if __name__ == "__main__":
    main()
