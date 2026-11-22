from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

input_file = Path("data/figure2/figure2_summary.csv")
output_dir = Path("plots/figure2")

output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)
df = df.sort_values("n")

fig, ax = plt.subplots(figsize=(11, 6))

ax.errorbar(
    df["n"],
    df["mean_regions"],
    yerr=df["std_regions"],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=6,
    capsize=4,
    label="Nasza replikacja (±1 odch. stand.)"
)

ax.set_title(
    "Model Axelroda – wpływ wielkości siatki na liczbę regionów kulturowych\n"
    "F = 5, q = 15",
    fontsize=13
)

ax.set_xlabel("Szerokość siatki (n)")
ax.set_ylabel("Średnia liczba stabilnych regionów")

ax.set_xticks(df["n"].tolist())
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

ax.grid(alpha=0.3)
ax.legend(loc="upper right")

fig.tight_layout()

output_file = output_dir / "figure2_mean_regions_std.png"

plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print("Wykres zapisano do:", output_file)

axelrod_n = [2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 50, 100]

axelrod_regions = [
    2.5, 9.5, 15.0, 18.5, 22.7,
    20.2, 17.2, 16.4, 14.8, 10.6,
    10.2, 10.2, 9.5, 5.6, 1.8
]

fig, ax = plt.subplots(figsize=(12, 6.5))

ax.plot(
    axelrod_n,
    axelrod_regions,
    marker="s",
    linestyle="--",
    linewidth=2,
    markersize=5,
    label="Axelrod (1997) – odczyt przybliżony"
)

ax.errorbar(
    df["n"],
    df["mean_regions"],
    yerr=df["std_regions"],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=6,
    capsize=4,
    label="Nasza replikacja (±1 odch. stand.)"
)

ax.set_title(
    "Rysunek 2 – porównanie wyników Axelroda z replikacją\n"
    "F = 5, q = 15",
    fontsize=14
)

ax.set_xlabel("Szerokość siatki (n)")
ax.set_ylabel("Średnia liczba stabilnych regionów")

ax.set_xlim(0, 105)
ax.set_ylim(bottom=0)

ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 80, 100])

ax.grid(alpha=0.3)
ax.legend(loc="upper right")

fig.tight_layout()

output_file = output_dir / "figure2_comparison.png"

plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print("Wykres porównawczy zapisano do:", output_file)
