"""
plot.py

Nástroj pro vykreslování a statistickou analýzu výsledků evolučního programu (GP) pro VRPTW:
  - plot_all_convergence                   – mřížka konvergenčních grafů (stín Min–Max + průměr)
  - plot_all_pareto                        – Pareto scatter grafy (vzdálenost vs. počet tras)
  - print_final_results                    – výpis souhrnu finálních výsledků z CSV
  - plot_boxplot_from_all_convergences     – boxplot fitness pro různé konfigurace
  - plot_all_from_directory                – porovnání GP vs. heuristiky Savings/Early‑First
  - plot_average_convergence_from_directory – průměrné konvergenční křivky napříč soubory
  - friedman_on_final_fitness              – Friedmanův test na rozložení finální fitness
  - boxplot_final_distances                – violin plot finálních vzdáleností
  - plot_algorithms_comparison             – boxploty porovnávající algoritmy podle velikosti instance

Závislosti:
  matplotlib, numpy, pandas, scipy.stats, glob, os, csv, re

Autor:      Petr Kaška
Vytvořeno:  2025-04-22
"""
from matplotlib.patches import Patch
import glob, os, csv, re, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import friedmanchisquare

def plot_all_convergence(directory="."):

    pattern = os.path.join(directory, "convergence_*.csv")
    csv_files = glob.glob(pattern)

    csv_files = csv_files[:9]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, csv_file in enumerate(csv_files):
        generations_data = {}
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                generation = int(row["generation"])
                fitness = float(row["best_fitness_in_generation"])
                generations_data.setdefault(generation, []).append(fitness)
        
        sorted_generations = sorted(generations_data.keys())
        gens, mins, maxs, avgs = [], [], [], []
        for g in sorted_generations:
            fits = generations_data[g]
            gens.append(g)
            mins.append(min(fits))
            maxs.append(max(fits))
            avgs.append(np.mean(fits))
        
        ax = axes[i]
        ax.fill_between(gens, mins, maxs, alpha=0.2, label="Min–Max")
        ax.plot(gens, avgs, label="Průměrná fitness")
        label = os.path.basename(csv_file).replace("convergence_", "").replace(".csv", "")
        ax.set_title(label)
        ax.set_xlabel("Generace")
        ax.set_ylabel("Fitness (nižší je lepší)")
        ax.legend()
        ax.grid(True)
    
    for j in range(i+1, len(axes)):
         axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()

def plot_all_pareto(directory="."):

    pattern = os.path.join(directory, "final_results_*.csv")
    csv_files = glob.glob(pattern)

    csv_files = csv_files[:9]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, csv_file in enumerate(csv_files):
        distances = []
        routes = []
        runs = []
        with open(csv_file, mode='r', encoding='utf-8') as ff:
            reader = csv.DictReader(ff)
            for row in reader:
                run_id = int(row["run"])
                total_dist = float(row["total_distance"])
                n_routes = float(row["number_of_routes"])
                runs.append(run_id)
                distances.append(total_dist)
                routes.append(n_routes)
        
        ax = axes[i]
        ax.scatter(distances, routes)

        for j, run_id in enumerate(runs):
            ax.annotate(str(run_id), (distances[j], routes[j]), fontsize=8)
        label = os.path.basename(csv_file).replace("final_results_", "").replace(".csv", "")
        ax.set_title(label)
        ax.set_xlabel("Celková vzdálenost")
        ax.set_ylabel("Počet tras")
        ax.grid(True)
    
    for j in range(i+1, len(axes)):
         axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()

def print_final_results(csv_file):
    
    with open(csv_file, mode='r', newline='') as ff:
        reader = csv.DictReader(ff)
        for row in reader:
            run_id = row["run"]
            final_fitness = row["final_fitness"]
            total_distance = row["total_distance"]
            number_of_routes = row["number_of_routes"]
            best_expr = row["best_expression"]
            
            print(f"\n=== Výsledky pro běh {run_id} ===")
            print(f"  Final fitness:    {final_fitness}")
            print(f"  Total distance:   {total_distance}")
            print(f"  Number of routes: {number_of_routes}")
            print(f"  Best expression:  {best_expr}")

def plot_boxplot_from_all_convergences(directory="."):

    pattern = os.path.join(directory, "convergence_*.csv")
    csv_files = glob.glob(pattern)

    data_for_boxplot = []
    labels = []

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        label = re.sub(r"^convergence_|\.csv$", "", filename)

        fitness_values = []
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fitness = float(row["best_fitness_in_generation"])
                fitness_values.append(fitness)

        data_for_boxplot.append(fitness_values)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(12, 6))

    custom_labels = [
        "Min=1, Max=10",
        "Min=1, Max=3",
        "Min=1, Max=4",
        "Min=1, Max=5",
        "Min=1, Max=6",
        "Min=1, Max=7",
        "Min=1, Max=8",
        "Min=1, Max=9",
        "Min=2, Max=10",
        "Min=2, Max=3",
        "Min=2, Max=4",
        "Min=2, Max=5",
        "Min=2, Max=6",
        "Min=2, Max=7",
        "Min=2, Max=8",
        "Min=2, Max=9",
        "Min=3, Max=10",
        "Min=3, Max=4",
        "Min=3, Max=5",
        "Min=3, Max=6",
        "Min=3, Max=7",
        "Min=3, Max=8",
        "Min=3, Max=9",
        "Min=4, Max=10",
        "Min=4, Max=5",
        "Min=4, Max=6",
        "Min=4, Max=7",
        "Min=4, Max=8",
        "Min=4, Max=9",
        "Min=5, Max=10",
        "Min=5, Max=6",
        "Min=5, Max=7",
        "Min=5, Max=8",
        "Min=5, Max=9",
        "Min=6, Max=10",
        "Min=6, Max=7",
        "Min=6, Max=8",
        "Min=6, Max=9",
        "Min=7, Max=10",
        "Min=7, Max=8",
        "Min=7, Max=9",

    ]

    bp = ax.boxplot(
        data_for_boxplot,
        labels=custom_labels,
        sym="",
        whis=1.5,
        patch_artist=True
    )

    for box in bp['boxes']:
        box.set_facecolor("#8fc9fb")

    ax.set_xlabel("Parametry běhů (populace=1000; genenrací=400; křížení=0.3; mutace=0.8; shrink=0.2; cooling_factor=0.995; velikost turnaje=12; elitismus=2)")
    ax.set_ylabel("Fitness (km)")
    ax.set_title("Zkoumání chování fitness při změně parametrů výšky stromu, který byl přidávám při mutaci (min a max hodnoty) ")

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()




def plot_all_from_directory(directory, heuristics=None):

    if heuristics is None:
        heuristics = [
            {'label': 'Saving',      'fitness': 1454.91393062},
            {'label': 'Early-First', 'fitness': 2994.27393248}
        ]
    linestyles = [':', '--']
    for idx, h in enumerate(heuristics):
        if idx == 0:
            h['color']     = 'tab:blue'
            h['linestyle'] = linestyles[idx]
        else:
            h['color']     = 'tab:red'
            h['linestyle'] = linestyles[idx]

    files = glob.glob(os.path.join(directory, "convergence_*.csv"))
    df    = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    stats = (df.groupby('generation')['best_fitness_in_generation']
               .agg(['mean','std'])
               .reset_index())
    gens, mean, std = stats['generation'], stats['mean'], stats['std']

    fig, ax1 = plt.subplots(figsize=(12,6))

    mean_line, = ax1.plot(
        gens, mean,
        color='tab:blue',
        label='GP průměr ±1σ'
    )
    ax1.fill_between(gens, mean-std, mean+std, color='tab:blue', alpha=0.2)
    ax1.set_xlabel('Generace')
    ax1.set_ylabel('Fitness GP a Saving (km)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    save_line = ax1.axhline(
        heuristics[0]['fitness'],
        color=heuristics[0]['color'],
        linestyle=heuristics[0]['linestyle'],
        label=f"{heuristics[0]['label']} {heuristics[0]['fitness']:.0f} km"
    )

    ax2 = ax1.twinx()
    early_line = ax2.axhline(
        heuristics[1]['fitness'],
        color=heuristics[1]['color'],
        linestyle=heuristics[1]['linestyle'],
        label=f"{heuristics[1]['label']} {heuristics[1]['fitness']:.0f} km"
    )
    ax2.set_ylabel('Fitness Early‑First (km)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    top = ax2.get_ylim()[1]
    ax2.set_ylim(1400, top)

    ax1.legend(
        handles=[mean_line, save_line, early_line],
        loc='upper right',
        bbox_to_anchor=(1, 0.85)
    )

    plt.title('Konvergenční křivka GP s Savings na levé a Early‑First na pravé ose')
    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.10)

    try:
        mngr = plt.get_current_fig_manager()
        mngr.window.showMaximized()
    except Exception:
        pass

    plt.show()

    if heuristics is None:
        heuristics = [
            {'label': 'Saving',      'fitness': 1454.91393062, 'vehicles': 19},
            {'label': 'Early-First', 'fitness': 2994.27393248, 'vehicles': 7}
        ]

    final_files = glob.glob(os.path.join(directory, "final_results*.csv"))
    final_df = pd.read_csv(final_files[0])

    fig, ax1 = plt.subplots(figsize=(6,6))
    ga_sc = ax1.scatter(
        final_df['number_of_routes'],
        final_df['final_fitness'],
        s=50, alpha=0.7,
        color='tab:blue',
        label='GA runs'
    )

    saving = heuristics[0]
    save_sc = ax1.scatter(
        saving['vehicles'],
        saving['fitness'],
        marker='X', s=100,
        color='tab:green',
        label=f"{saving['label']} {saving['fitness']:.0f} km, vozidel {saving['vehicles']}"
    )

    ax1.set_xlabel('Počet tras (vozidel)')
    ax1.set_ylabel('Fitness GP a Saving (km)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    early = heuristics[1]
    ax2 = ax1.twinx()
    early_sc = ax2.scatter(
        early['vehicles'],
        early['fitness'],
        marker='X', s=100,
        color='tab:red',
        label=f"{early['label']} {early['fitness']:.0f} km, vozidel {early['vehicles']}"
    )
    ax2.set_ylabel('Fitness Early‑First (km)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(1400, ax2.get_ylim()[1])

    handles = [ga_sc, save_sc, early_sc]
    labels  = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper right')

    plt.title('Pareto front: GP vs. Saving vs. Early‑First')
    plt.tight_layout()
    plt.show()


def plot_average_convergence_from_directory(directory="."):

    pattern = os.path.join(directory, "convergence_*.csv")
    csv_files = sorted(glob.glob(pattern))

    fig, ax = plt.subplots(figsize=(10,6))
    for csv_file in csv_files:
        fname = os.path.basename(csv_file)
        label = re.sub(r"^convergence_|\.csv$", "", fname)

        df = pd.read_csv(csv_file, usecols=["generation", "best_fitness_in_generation"])
        mean_df = (
            df
            .groupby("generation")["best_fitness_in_generation"]
            .mean()
            .reset_index()
            .rename(columns={"best_fitness_in_generation": "mean_fitness"})
        )

        ax.plot(
            mean_df["generation"],
            mean_df["mean_fitness"],
            linewidth=2,
            label=label
        )

    ax.set_xlabel("Generace")
    ax.set_ylabel("Průměrná Fitness (km)")
    ax.set_title("Průměrné konvergenční křivky pro každý soubor")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()


def friedman_on_final_fitness(directory="."):
    paths = sorted(glob.glob(os.path.join(directory, "convergence_*.csv")))

    runs = []
    configs = []

    for path in paths:
        name = os.path.basename(path)
        if "Always choose the best" in name:
            cfg = "always_best"
        else:
            m = re.search(r"sigma=([0-9.]+)", name)
            cfg = f"sigma={m.group(1)}" if m else name

        df = pd.read_csv(path)
        last_gen = df["generation"].max()
        last = df[df["generation"] == last_gen][["run", "best_fitness_in_generation"]].copy()
        last["config"] = cfg
        runs.append(last)
        configs.append(cfg)

    all_df = pd.concat(runs, ignore_index=True)
    pivot = all_df.pivot(index="run", columns="config", values="best_fitness_in_generation")
    print("Data připravená pro Friedmanův test (rows=runs, cols=configs):")
    print(pivot.head())

    arrays = [pivot[c].values for c in pivot.columns]
    stat, p = friedmanchisquare(*arrays)
    print(f"\nFriedman χ² = {stat:.3f}, p = {p:.4f}")

    k = len(pivot.columns)
    n = len(pivot)
    W = stat / (n * (k-1))
    print(f"Kendallovo W = {W:.3f}")


def boxplot_final_distances(directory="."):
    variants = ["Always choose the best", "σ=0.5", "σ=1", "σ=2", "σ=30"]
    data = {}
    for v in variants:
        df = pd.read_csv(f"{directory}final_results_{v}.csv")
        data[v] = df["total_distance"]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.violinplot([data[v] for v in variants], showmeans=True)
    ax.set_xticks([1,2,3,4,5]); ax.set_xticklabels(variants)
    ax.set_ylabel("Celková vzdálenost (km)")
    ax.set_title("Violin plot - finálních fitness")
    plt.tight_layout()
    plt.show()


def plot_algorithms_comparison(data_dir="."):
    custom_labels =["25", "50", "100"]
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    dfs = []
    for f in files:
        alg = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        df["alg"] = alg
        df["size"] = df["size"].astype(int)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    sizes = sorted(data["size"].unique())
    algs = data["alg"].unique().tolist()
    n_algs = len(algs)
    x = np.arange(len(sizes))
    width = 0.8 / n_algs

    colors = plt.cm.tab10.colors[:n_algs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    axes[1].set_xlabel("Velikost instance (počet zákazníků)")
    axes[0].set_xlabel("Velikost instance (počet zákazníků)")


    ax = axes[0]
    for i, alg in enumerate(algs):
        vals = [data[(data["alg"] == alg) & (data["size"] == s)]["num_routes"] for s in sizes]
        pos = x - 0.4 + width/2 + i*width
        bp = ax.boxplot(vals, positions=pos, widths=width*0.9,
                        patch_artist=True, showfliers=False)
        for box in bp['boxes']:
            box.set_facecolor(colors[i])
            box.set_alpha(0.6)
    ax.set_title("Rozdělení počtu tras")
    ax.set_ylabel("Počet tras")
    ax.set_xticks(x)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(custom_labels)
    

    ax = axes[1]
    for i, alg in enumerate(algs):
        vals = [data[(data["alg"] == alg) & (data["size"] == s)]["total_distance"] for s in sizes]
        pos = x - 0.4 + width/2 + i*width
        bp = ax.boxplot(vals, positions=pos, widths=width*0.9,
                        patch_artist=True, showfliers=False)
        for box in bp['boxes']:
            box.set_facecolor(colors[i])
            box.set_alpha(0.6)
    ax.set_title("Rozdělení ujeté vzdálenosti")
    ax.set_ylabel("Celková vzdálenost (km)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:03d}" for s in sizes])
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend_handles = [Patch(facecolor=colors[i], label=algs[i], alpha=0.6) for i in range(n_algs)]
    axes[1].legend(handles=legend_handles, title="Algoritmus", loc="upper left", bbox_to_anchor=(1,1))
    custom_labels = ["25", "50", "100"]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(custom_labels)


    plt.tight_layout()
    plt.show()