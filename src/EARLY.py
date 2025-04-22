"""
EARLY.py

Greedy algoritmus pro řešení problému VRPTW („earliest‑first“):
  - distance                         – výpočet euklidovské vzdálenosti
  - construct_solution_earliest_first – sestavení tras podle nejčasnějšího začátku
  - batch_earliest                   – dávkové spuštění nad Solomon XML instancemi
  - main                             – vstupní bod: načtení argumentů a volání batch_earliest

Závislosti:
  math, copy, glob, os, csv, sys, parse_solomon

Author:      Petr Kaška
Created:     2025-04-22
"""
import math
import copy
import glob
import os
import csv
import sys

from parse_solomon import parse_solomon_xml

def distance(x1, y1, x2, y2):
    """
    Vrátí euklidovskou vzdálenost mezi dvěma body (x1,y1) a (x2,y2),
    vypočtenou pomocí math.hypot.
    """
    return math.hypot(x2-x1, y2-y1)

def construct_solution_earliest_first(instance):
    """
    Greedý algoritmus pro VRPTW: vždy vybere dalšího zákazníka
    podle nejmenšího začátku jeho časového okna (tw_start),
    pokud je cesta k němu kapacitně a časově (okno tw_end) možná.
    
    Parametry instance:
      - instance['customers']: seznam zákazníků (dict s klíči id, x, y, demand, tw_start, tw_end, service_time)
      - instance['depot']: dict s klíči x, y
      - instance['vehicle_capacity']: kapacita vozidla
      - instance['vehicle_speed']: rychlost vozidla
    
    Vrací:
      - all_routes: seznam cest; každá cesta je seznam dvojic (customer_id, distance_from_previous)
      - total_dist: celková délka všech tras
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
        feasible = []
        for c in customers:
            if current_load + c['demand'] > cap:
                continue
            d       = distance(last_x, last_y, c['x'], c['y'])
            arrival = current_time + d/speed
            if arrival > c['tw_end']:
                continue
            feasible.append((c, d, arrival))

        if not feasible:
            if current_route:
                d_back = distance(last_x, last_y, depot['x'], depot['y'])
                current_route.append(("Depot", d_back))
                all_routes.append(current_route)
            if not customers:
                break
            current_route = []
            current_load  = 0.0
            current_time  = 0.0
            last_x, last_y = depot['x'], depot['y']
            continue

        c, d, arrival = min(feasible, key=lambda tup: tup[0]['tw_start'])
        start  = max(arrival, c['tw_start'])
        finish = start + c['service_time']

        current_route.append((c['id'], d))
        current_load  += c['demand']
        current_time   = finish
        last_x, last_y = c['x'], c['y']
        customers.remove(c)

    total_dist = sum(d for route in all_routes for (_, d) in route)
    return all_routes, total_dist

def batch_earliest(inst_dir, out_csv="results_by_size_earliest.csv"):
    """
    Pro každý soubor R2*_*.xml v adresáři inst_dir:
      1) načte instanci pomocí parse_solomon_xml
      2) vygeneruje řešení greedy earliest-first
      3) zapíše do out_csv řádky [size, total_distance, num_routes]
    """
    xmls = sorted(glob.glob(os.path.join(inst_dir, "R2*_*.xml")))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "total_distance", "num_routes"])
        for xml in xmls:
            size = os.path.basename(xml).split("_")[1].replace(".xml", "")
            instance = parse_solomon_xml(xml)
            routes, dist = construct_solution_earliest_first(instance)
            num_routes = len(routes)
            writer.writerow([size, f"{dist:.2f}", num_routes])
            print(f"{xml:30s} size={size}  dist={dist:.2f}  routes={num_routes}")

def main():
    """
    Vstupní bod skriptu: očekává dva argumenty
      1) adresář s XML instancemi
      2) cesta k výstupnímu CSV
    """

    inst_dir = sys.argv[1]
    out_csv  = sys.argv[2]
    batch_earliest(inst_dir, out_csv)

if __name__ == "__main__":
    main()
