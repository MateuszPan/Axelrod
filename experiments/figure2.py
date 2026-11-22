import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from model import (
    create_grid,
    run_simulation_optimized,
    count_regions,
    is_stable
)

F = 5
q = 15

n_values = [2, 5, 10, 12, 15, 20, 25, 30, 35, 50, 100]
repetitions = 40
large_repetitions = 10
max_steps = 3000000000
workers = 4

output_dir = Path("data/figure2")
output_file = output_dir / "figure2_runs.csv"

fieldnames = [
    "run", "n", "F", "q",
    "seed_grid", "seed_interactions",
    "steps", "changes", "stable", "regions"
]


def simulate(task):
    n, run = task

    experiment_id = n * 1000 + run
    seed_grid = 100000 + experiment_id * 2
    seed_interactions = seed_grid + 1

    rng_grid = np.random.default_rng(seed_grid)
    rng_interactions = np.random.default_rng(seed_interactions)

    grid = create_grid(n, F, q, rng=rng_grid)

    result = run_simulation_optimized(
        grid,
        rng_interactions,
        max_steps=max_steps
    )

    stable = is_stable(result["grid"])

    if not stable:
        raise RuntimeError(
            f"Nie osiągnięto stabilności: n={n}, run={run}, "
            f"steps={result['steps']}"
        )

    regions = count_regions(result["grid"])

    return {
        "run": run,
        "n": n,
        "F": F,
        "q": q,
        "seed_grid": seed_grid,
        "seed_interactions": seed_interactions,
        "steps": result["steps"],
        "changes": result["changes"],
        "stable": stable,
        "regions": regions
    }


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = set()

    if output_file.exists():
        with open(output_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != fieldnames:
                raise RuntimeError("Nieprawidłowa struktura istniejącego CSV.")

            for row in reader:
                if row["stable"] != "True":
                    raise RuntimeError("W pliku CSV znajduje się niestabilny przebieg.")

                key = (int(row["n"]), int(row["run"]))

                if key in completed:
                    raise RuntimeError("Wykryto powtórzony przebieg w CSV.")

                completed.add(key)

    tasks = [
        (n, run)
        for n in n_values
        for run in range(
            1,
            (large_repetitions if n >= 50 else repetitions) + 1
        )
        if (n, run) not in completed
    ]

    print("Ukończone wcześniej:", len(completed))
    print("Pozostałe do wykonania:", len(tasks))
    print("Liczba procesów:", workers)

    if not tasks:
        print("Wszystkie zaplanowane symulacje są już ukończone.")
        return

    file_exists = output_file.exists()
    failed = []

    with open(output_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
            file.flush()

        if workers == 1:
            for task in tasks:
                row = simulate(task)
                writer.writerow(row)
                file.flush()

                print(
                    f"n={row['n']} | run={row['run']}/{large_repetitions if row['n'] >= 50 else repetitions} | "
                    f"steps={row['steps']} | "
                    f"changes={row['changes']} | "
                    f"regions={row['regions']}",
                    flush=True
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(simulate, task): task
                    for task in tasks
                }

                for future in as_completed(futures):
                    task = futures[future]

                    try:
                        row = future.result()
                    except RuntimeError as error:
                        failed.append(task)
                        print(
                            f"Nieukończony przebieg: n={task[0]}, run={task[1]} | {error}",
                            flush=True
                        )
                        continue

                    writer.writerow(row)
                    file.flush()

                    print(
                        f"n={row['n']} | run={row['run']}/{large_repetitions if row['n'] >= 50 else repetitions} | "
                        f"steps={row['steps']} | "
                        f"changes={row['changes']} | "
                        f"regions={row['regions']}",
                        flush=True
                    )

    print()

    if failed:
        print("Nie wszystkie symulacje osiągnęły stabilność.")
        print("Nieukończone przebiegi:", failed)
        print("Ukończone wyniki zapisano do:", output_file)
    else:
        print("Zakończono zaplanowane symulacje.")
        print("Wyniki zapisano do:", output_file)


if __name__ == "__main__":
    main()
