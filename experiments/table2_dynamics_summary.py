from pathlib import Path
import pandas as pd

input_file = Path("data/table2/table2_runs.csv")
output_file = Path("data/table2/table2_dynamics_summary.csv")

df = pd.read_csv(input_file)

if len(df) != 90 or not df["stable"].all():
    raise ValueError("Niekompletne lub niestabilne symulacje.")

summary = df.groupby(["F", "q"]).agg(
    runs=("run", "count"),
    mean_steps=("steps", "mean"),
    std_steps=("steps", "std"),
    mean_changes=("changes", "mean"),
    std_changes=("changes", "std"),
    total_steps=("steps", "sum"),
    total_changes=("changes", "sum")
).reset_index()

if len(summary) != 9 or not (summary["runs"] == 10).all():
    raise ValueError("Nieprawidłowa liczba kombinacji lub powtórzeń.")

summary["change_rate_pct"] = (
    100 * summary["total_changes"] / summary["total_steps"]
)

summary = summary.drop(columns=["total_steps", "total_changes"])
summary = summary.round(4)

summary.to_csv(output_file, index=False, encoding="utf-8")

print(summary.to_string(index=False))
print("\nWyniki zapisano do:", output_file)