import csv
import numpy as np

from model import create_grid, run_simulation_optimized, count_regions, is_stable

with open("data/figure2/figure2_runs.csv", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

row = next(row for row in rows if int(row["n"]) == 10 and int(row["run"]) == 1)

n = int(row["n"])
F = int(row["F"])
q = int(row["q"])

rng_grid = np.random.default_rng(int(row["seed_grid"]))
rng_interactions = np.random.default_rng(int(row["seed_interactions"]))

grid = create_grid(n, F, q, rng=rng_grid)

result = run_simulation_optimized(
    grid,
    rng_interactions,
    max_steps=10000000
)

checks = {
    "steps": result["steps"] == int(row["steps"]),
    "changes": result["changes"] == int(row["changes"]),
    "regions": count_regions(result["grid"]) == int(row["regions"]),
    "stable": is_stable(result["grid"]) == (row["stable"] == "True")
}

for name, passed in checks.items():
    print(f"{name}: {'OK' if passed else 'ROZNICA'}")

if all(checks.values()):
    print("Odtworzono identyczny wynik symulacji.")
else:
    print("Wykryto różnice między środowiskami.")
