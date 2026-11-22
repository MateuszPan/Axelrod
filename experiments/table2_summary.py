import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

input_file = Path("data/table2/table2_runs.csv")
output_file = Path("data/table2/table2_summary.csv")

groups = defaultdict(list)
completed = set()

with open(input_file, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        key = (int(row["F"]), int(row["q"]))
        run_key = (key[0], key[1], int(row["run"]))

        if row["stable"] != "True":
            raise ValueError("Znaleziono niestabilną symulację.")

        if run_key in completed:
            raise ValueError("Znaleziono powtórzony wynik symulacji.")

        completed.add(run_key)
        groups[key].append(int(row["regions"]))

if len(groups) != 9 or len(completed) != 90:
    raise ValueError("Niekompletne wyniki eksperymentu.")

for values in groups.values():
    if len(values) != 10:
        raise ValueError("Nieprawidłowa liczba powtórzeń.")

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "F", "q", "runs", "mean_regions",
        "std_regions", "min_regions", "max_regions"
    ])

    for F, q in sorted(groups):
        values = groups[(F, q)]

        writer.writerow([
            F,
            q,
            len(values),
            round(mean(values), 4),
            round(stdev(values), 4),
            min(values),
            max(values)
        ])

        print(
            "F =", F,
            "| q =", q,
            "| Średnia =", round(mean(values), 4),
            "| Odchylenie =", round(stdev(values), 4)
        )

print("\nPodsumowanie zapisano do:", output_file)