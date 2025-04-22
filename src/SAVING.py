"""
SAVING.py

Nástroje pro řešení problému VRPTW pomocí Clarke–Wright Savings heuristiky:
  - distance                  – výpočet euklidovské vzdálenosti
  - check_route_feasibility   – ověření trasy z hlediska časových oken a kapacity
  - construct_solution_saving_heuristic
                              – konstrukce řešení savings heuristikou
  - batch_saving              – dávkové zpracování Solomonových XML instancí a výstup do CSV

Author:      Petr Kaška
Created:     2025-04-22
"""
import math
import copy
import glob
import os
import csv

from parse_solomon import parse_solomon_xml

def distance(x1, y1, x2, y2):
    """
    Spočte euklidovskou vzdálenost mezi dvěma body (x1,y1) a (x2,y2).
    """
    return math.hypot(x2-x1, y2-y1)

def check_route_feasibility(route, instance):
    """
    Ověří, zda je daná trasa (seznam zákazníků) v instance splnitelná:
      - Zda se dodrží časová okna (tw_start, tw_end)
      - Zda se nepřekročí kapacita vozidla
    Počítá postupně:
      * vzdálenost a čas příjezdu ke každému zákazníkovi
      * případné čekání do začátku okna
      * servisní čas
      * akumulaci nákladu
    Na konci zahrne návrat do depa.
    Vrátí trojici:
      (feasible: bool, finish_time: float, total_distance: float)
    Pokud trasa nevyhovuje, vrací (False, None, None).
    """
    depot = instance['depot']
    speed = instance['vehicle_speed']
    capacity = instance['vehicle_capacity']
    current_time = 0.0
    current_load = 0.0
    total_distance = 0.0
    last_x, last_y = depot['x'], depot['y']
    for customer in route:
        d = distance(last_x, last_y, customer['x'], customer['y'])
        arrival = current_time + d/speed
        start = max(arrival, customer['tw_start'])
        if start > customer['tw_end']:
            return False, None, None
        current_time = start + customer['service_time']
        current_load += customer['demand']
        total_distance += d
        last_x, last_y = customer['x'], customer['y']

    d_back = distance(last_x, last_y, depot['x'], depot['y'])
    total_distance += d_back
    if current_load > capacity:
        return False, None, None
    return True, current_time, total_distance

def construct_solution_saving_heuristic(instance):
    """
    Clarke-Wright Savings heuristika pro VRPTW:
      1. Inicializace: každý zákazník ve vlastní trase
      2. Spočítání savings pro každou dvojici (i,j)
      3. Setřídění savings sestupně
      4. Procházení savings: pokud lze bez porušení časových oken a kapacity
         spojit trasu končící na i s trasou začínající na j, provede se merge
      5. Výsledkem je množina tras a celková vzdálenost.

    Vrací:
      - all_routes: list of routes, kde route je seznam zákazníků (dict)
      - total_dist: součet délek všech tras (včetně návratů do depa)
    """
    depot = instance['depot']
    customers = copy.deepcopy(instance['customers'])
    routes = {c['id']:[c] for c in customers}
    assignment = {c['id']:c['id'] for c in customers}
    savings = []
    for i in customers:
        for j in customers:
            if i['id']==j['id']:
                continue
            if assignment[i['id']]!=assignment[j['id']]:
                s = (
                    distance(depot['x'], depot['y'], i['x'], i['y']) +
                    distance(depot['x'], depot['y'], j['x'], j['y']) -
                    distance(i['x'], i['y'], j['x'], j['y'])
                )
                savings.append((i['id'], j['id'], s))
    savings.sort(key=lambda x:x[2], reverse=True)
    for i_id, j_id, _ in savings:
        ri, rj = assignment[i_id], assignment[j_id]
        if ri == rj:
            continue
        route_i, route_j = routes[ri], routes[rj]
        if route_i[-1]['id'] == i_id and route_j[0]['id'] == j_id:
            merged = route_i + route_j
            feas, _, _ = check_route_feasibility(merged, instance)
            if feas:
                routes[ri] = merged
                del routes[rj]
                for c in merged:
                    assignment[c['id']] = ri

    all_routes = []
    total_dist = 0.0
    for route in routes.values():
        last_x, last_y = depot['x'], depot['y']
        rd = 0.0
        for c in route:
            d = distance(last_x, last_y, c['x'], c['y'])
            rd += d
            last_x, last_y = c['x'], c['y']
        d_back = distance(last_x, last_y, depot['x'], depot['y'])
        rd += d_back
        all_routes.append(route)
        total_dist += rd
    return all_routes, total_dist

def batch_saving(inst_dir, out_csv="saving_results.csv"):
    """
    Pro všechny Solomonovy XML instance v adresáři inst_dir:
      1) načte instanci
      2) vygeneruje řešení saving heuristikou
      3) zapíše do out_csv řádky [size, total_distance, num_routes]
    """
    xmls = sorted(glob.glob(os.path.join(inst_dir, "R2*_*.xml")))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["size", "total_distance", "num_routes"])
        for xml in xmls:
            size = os.path.basename(xml).split("_")[1].replace(".xml", "")
            instance = parse_solomon_xml(xml)
            all_routes, dist = construct_solution_saving_heuristic(instance)
            num_routes = len(all_routes)
            w.writerow([size, f"{dist:.2f}", num_routes])
            print(f"{xml:30s}  size={size}  dist={dist:.2f}, routes={num_routes}")

if __name__ == "__main__":
    inst_dir = "data"
    batch_saving(inst_dir, out_csv="results_by_size.csv")