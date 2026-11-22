import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from model import create_grid, run_simulation, is_stable, count_regions

n = 3
F = 5
q = 3
seed = 123

rng_grid = np.random.default_rng(seed)
rng_interactions = np.random.default_rng(seed + 1)

grid = create_grid(n, F, q, rng=rng_grid)
initial_grid = grid.copy()

result = run_simulation(
    grid,
    rng_interactions,
    max_steps=100000
)

final_grid = result["grid"]
regions = count_regions(final_grid)
print("Stan początkowy:")
print(initial_grid)

print("\nStan końcowy:")
print(final_grid)

print("\nLiczba prób:", result["steps"])
print("Liczba zmian:", result["changes"])
print("Czy siatka jest stabilna?", is_stable(final_grid))
print("Liczba stabilnych regionów:", regions)

data_dir = Path("tests/data")
data_dir.mkdir(parents=True, exist_ok=True)

result_data = {
    "seed": seed,
    "n": n,
    "F": F,
    "q": q,
    "steps": result["steps"],
    "changes": result["changes"],
    "stable": bool(is_stable(final_grid)),
    "regions": regions
}

df = pd.DataFrame([result_data])

csv_path = data_dir / f"simulation_n{n}_F{F}_q{q}_seed{seed}.csv"

df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("Wyniki zapisano do:", csv_path)
