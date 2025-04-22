# VRPTW Genetic Programming Toolbox

This repository contains a suite of Python tools and scripts for solving the Vehicle Routing Problem with Time Windows (VRPTW) using Genetic Programming and heuristic methods. It includes parsers for Solomon XML instances, GP core implementations, heuristics, visualization tools, and utilities for managing experiments and results.

## Features

- **Solomon XML parser** for VRPTW instances  
- **Heuristic methods**: Clarke–Wright savings heuristic and earliest‑first greedy algorithm  
- **Genetic Programming core** with DEAP, custom primitives, and fitness functions (including Lagrange penalties)  
- **Experiment management**: parameterized CSV filenames, batch runs (sequential and parallel)  
- **Best‑tree extraction**: select best GP tree from CSV logs and apply to new instances  
- **Visualization & analysis**: convergence plots, Pareto fronts, boxplots, statistical tests (Friedman, Kendall’s W)  

## Prerequisites

- Python **3.8+**  
- **DEAP** library  
- **numpy**, **pandas**, **matplotlib**, **scipy**  

Install dependencies:

```bash
pip install deap numpy pandas matplotlib scipy
```

## /Src  
```
parse_solomon.py - Solomon XML parser
tree_operators.py - Safe arithmetic primitives for GP
SAVING.py - Clarke–Wright savings heuristic
EARLY.py - Greedy earliest‑first heuristic
VRPTW.py - DEAP GP setup and run logic
run_best_heuristic.py - Select and apply the best GP tree
plot.py -  Plotting and statistical analysis
run_multiple_times.py - Experiment orchestration (run/plot modes)
```


## Running code
### EARLY.py + SAVING.py
Run: ``python3 {SAVING.py,EARLY.py} ../data results.csv``. 

This way we run SAVING.py and EARLY.py on the whole dataset and the output will be “result.csv” which will contain three columns. After the row “size”, indicating the size of the instance on which the algorithm was run (025/050/100). Then “dist”, indicating the distance found by the algorithm, and the third ‘routes’, the total number of routes found by the algorithm at distance “dist”.

### run_multiple_times.py
Run: ``python3 run_multiple_times.py {run/runParallel/plot}``

This is the main script of my program. Its purpose is to comprehensively orchestrate GP-experiments over VRPTW instances, collect results into CSV, and generate statistics and graphs. It works in three modes 1)plot: only parsing and plotting graphs from existing CSVs.
2)run: sequential execution of NP runs of GP and writing the results.
3)runParallel: parallel execution of NP runs of GP.
When the "plot" switch is specified, it is up to the user to import custom functions from the "plot.py" library into the code, depending on which statistics they want to track. The “run” and ‘runParallel’ switches do exactly the same thing, except that “runParallel” runs the individual runs in parallel, so the result is faster, but you need to check the resources available on the HW on which the script is running beforehand, it could also lead to extreme increases in the script's runtime. Finally, the proper variables are defined in the main function, which are related to running the script and testing the monitored properties. 


