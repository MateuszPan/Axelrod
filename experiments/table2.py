import csv
from pathlib import Path
import numpy as np

from model import (
    create_grid,
    run_simulation_optimized,
    count_regions,
    is_stable
)

n = 10
F_values = [5, 10, 15]
q_values = [5, 10, 15]
repetitions = 10

output_dir = Path("data/table2")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "table2_runs.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "run", "n", "F", "q",
        "seed_grid", "seed_interactions",
        "steps", "changes", "stable", "regions"
    ])

    for F in F_values:
        for q in q_values:
            print("\nParametry: F =", F, "| q =", q)

            for run in range(1, repetitions + 1):
                experiment_id = (
                    (F_values.index(F) * len(q_values) + q_values.index(q))
                    * repetitions + run - 1
                )

                seed_grid = 123 + experiment_id * 2
                seed_interactions = seed_grid + 1

                rng_grid = np.random.default_rng(seed_grid)
                rng_interactions = np.random.default_rng(seed_interactions)

                grid = create_grid(n, F, q, rng=rng_grid)

                result = run_simulation_optimized(
                    grid,
                    rng_interactions,
                    max_steps=1000000
                )

                regions = count_regions(result["grid"])
                stable = is_stable(result["grid"])

                writer.writerow([
                    run, n, F, q,
                    seed_grid, seed_interactions,
                    result["steps"], result["changes"],
                    stable, regions
                ])
                file.flush()

                print(
                    "Powtórzenie:", run,
                    "| Próby:", result["steps"],
                    "| Zmiany:", result["changes"],
                    "| Stabilność:", stable,
                    "| Regiony:", regions
                )

print("\nWyniki zapisano do:", output_file)