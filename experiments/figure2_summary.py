from pathlib import Path
import pandas as pd

input_file = Path("data/figure2/figure2_runs.csv")
output_file = Path("data/figure2/figure2_summary.csv")

df = pd.read_csv(input_file)

required_columns = ["n", "F", "q", "run", "regions", "steps", "changes", "stable"]

if not all(column in df.columns for column in required_columns):
    raise RuntimeError("W pliku CSV brakuje wymaganych kolumn.")

if not df["stable"].all():
    raise RuntimeError("W pliku znajdują się niestabilne symulacje.")

if df.duplicated(subset=["n", "run"]).any():
    raise RuntimeError("Wykryto powtórzone symulacje.")

if not ((df["F"] == 5) & (df["q"] == 15)).all():
    raise RuntimeError("Nieprawidłowe parametry F lub q.")

summary = (
    df.groupby("n")
    .agg(
        runs=("regions", "count"),
        mean_regions=("regions", "mean"),
        std_regions=("regions", "std"),
        min_regions=("regions", "min"),
        max_regions=("regions", "max"),
        mean_steps=("steps", "mean"),
        mean_changes=("changes", "mean")
    )
    .reset_index()
    .sort_values("n")
)

summary["std_regions"] = summary["std_regions"].fillna(0)

summary.to_csv(output_file, index=False)

print(summary.to_string(index=False))
print()
print("Liczba ukończonych symulacji:", len(df))
print("Podsumowanie zapisano do:", output_file)
