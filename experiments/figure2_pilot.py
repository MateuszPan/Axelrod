import csv
from pathlib import Path
import numpy as np

from model import (
    create_grid,
    run_simulation_optimized,
    count_regions,
    is_stable
)

n = 5
F = 5
q = 15

seed_grid = 123
seed_interactions = 124

rng_grid = np.random.default_rng(seed_grid)
rng_interactions = np.random.default_rng(seed_interactions)

grid = create_grid(n, F, q, rng=rng_grid)

result = run_simulation_optimized(
    grid,
    rng_interactions,
    max_steps=1000000
)

stable = is_stable(result["grid"])
regions = count_regions(result["grid"])

if not stable:
    raise RuntimeError("Symulacja nie osiągnęła stabilności.")

output_dir = Path("data/figure2")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "figure2_runs.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "run", "n", "F", "q",
        "seed_grid", "seed_interactions",
        "steps", "changes", "stable", "regions"
    ])

    writer.writerow([
        1, n, F, q,
        seed_grid, seed_interactions,
        result["steps"], result["changes"],
        stable, regions
    ])

print("Rozmiar siatki:", n, "x", n)
print("Liczba cech F:", F)
print("Liczba wartości cechy q:", q)
print("Liczba prób:", result["steps"])
print("Liczba zmian:", result["changes"])
print("Czy siatka jest stabilna?", stable)
print("Liczba stabilnych regionów:", regions)
print("Wyniki zapisano do:", output_file)