from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_file = Path("data/table2/table2_summary.csv")
output_dir = Path("plots/table2")
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(input_file)

plt.figure(figsize=(9, 6))

for F in sorted(df["F"].unique()):
    subset = df[df["F"] == F].sort_values("q")

    plt.plot(
        subset["q"],
        subset["mean_regions"],
        marker="o",
        linewidth=2,
        label=f"F = {F}"
    )

plt.title(
    "Model Axelroda – średnia liczba stabilnych regionów kulturowych\n"
    "Siatka 10 × 10, 10 powtórzeń dla każdej kombinacji parametrów",
    fontsize=13,
    pad=15
)
plt.xlabel("Liczba możliwych wartości cechy (q)")
plt.ylabel("Średnia liczba stabilnych regionów")
plt.xticks([5, 10, 15])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_file = output_dir / "table2_mean_regions.png"
plt.savefig(output_file, dpi=300)
plt.show()

print("Wykres zapisano do:", output_file)

plt.figure(figsize=(10, 6))

for F in sorted(df["F"].unique()):
    subset = df[df["F"] == F].sort_values("q")

    plt.errorbar(
        subset["q"],
        subset["mean_regions"],
        yerr=subset["std_regions"],
        marker="o",
        capsize=5,
        linewidth=2,
        label=f"F = {F}"
    )

plt.title(
    "Model Axelroda – średnia liczba stabilnych regionów kulturowych\n"
    "Siatka 10 × 10, 10 powtórzeń, słupki błędów: ±1 odchylenie standardowe",
    fontsize=12,
    pad=15
)

plt.xlabel("Liczba możliwych wartości cechy (q)")
plt.ylabel("Średnia liczba stabilnych regionów")
plt.xticks([5, 10, 15])
plt.ylim(bottom=0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_file_std = output_dir / "table2_mean_regions_std.png"
plt.savefig(output_file_std, dpi=300)
plt.show()

print("Wykres zapisano do:", output_file_std)

runs_df = pd.read_csv("data/table2/table2_runs.csv")

groups = []
labels = []

for F in [5, 10, 15]:
    for q in [5, 10, 15]:
        subset = runs_df[
            (runs_df["F"] == F) & (runs_df["q"] == q)
        ].sort_values("run")

        groups.append(subset["regions"].to_numpy())
        labels.append(f"F={F}\nq={q}")

fig, ax = plt.subplots(figsize=(12, 7))

positions = list(range(1, 10))

ax.boxplot(
    groups,
    positions=positions,
    widths=0.5,
    patch_artist=False,
    showfliers=False
)

for position, values in zip(positions, groups):
    offsets = np.linspace(-0.15, 0.15, len(values))

    ax.scatter(
        position + offsets,
        values,
        alpha=0.75,
        s=30,
        zorder=3
    )

ax.set_title(
    "Model Axelroda – liczba stabilnych regionów kulturowych\n"
    "Wyniki pojedynczych symulacji pogrupowane według F i q\n"
    "Siatka 10 × 10, 10 powtórzeń dla każdej kombinacji parametrów",
    fontsize=12,
    pad=15
)

ax.set_xlabel("Kombinacje parametrów F i q")
ax.set_ylabel("Liczba stabilnych regionów")
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

output_file_runs = output_dir / "table2_individual_regions.png"
fig.savefig(output_file_runs, dpi=300)
plt.show()

print("Wykres zapisano do:", output_file_runs)

dynamics_df = runs_df.groupby(["F", "q"], as_index=False).agg(
    mean_steps=("steps", "mean"),
    mean_changes=("changes", "mean")
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for F in [5, 10, 15]:
    subset = dynamics_df[dynamics_df["F"] == F].sort_values("q")

    axes[0].plot(
        subset["q"],
        subset["mean_steps"],
        marker="o",
        linewidth=2,
        label=f"F = {F}"
    )

    axes[1].plot(
        subset["q"],
        subset["mean_changes"],
        marker="o",
        linewidth=2,
        label=f"F = {F}"
    )

axes[0].set_title("Średnia liczba prób interakcji")
axes[0].set_ylabel("Liczba prób")

axes[1].set_title("Średnia liczba zmian kulturowych")
axes[1].set_ylabel("Liczba zmian")

for ax in axes:
    ax.set_xlabel("Liczba możliwych wartości cechy (q)")
    ax.set_xticks([5, 10, 15])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(
    "Model Axelroda – dynamika osiągania stabilności\n"
    "Siatka 10 × 10, średnie z 10 powtórzeń dla każdej kombinacji F i q",
    fontsize=13
)

fig.tight_layout()

output_file_dynamics = output_dir / "table2_dynamics.png"
fig.savefig(output_file_dynamics, dpi=300)
plt.show()

print("Wykres zapisano do:", output_file_dynamics)

axelrod_values = {
    5: [1.0, 3.2, 20.0],
    10: [1.0, 1.0, 1.4],
    15: [1.0, 1.0, 1.2]
}

comparison_df = pd.read_csv("data/table2/table2_summary.csv")

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)

q_values = [5, 10, 15]

for ax, F in zip(axes, [5, 10, 15]):
    subset = comparison_df[comparison_df["F"] == F].set_index("q")
    subset = subset.loc[q_values]

    ax.plot(
        q_values,
        axelrod_values[F],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Axelrod (1997)"
    )

    ax.errorbar(
        q_values,
        subset["mean_regions"],
        yerr=subset["std_regions"],
        marker="o",
        linestyle="-",
        linewidth=2,
        capsize=5,
        label="Nasza replikacja ± SD"
    )

    ax.set_title(f"F = {F}")
    ax.set_xlabel("Liczba wartości cechy (q)")
    ax.set_xticks(q_values)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)

axes[0].set_ylabel("Średnia liczba stabilnych regionów")

axes[0].legend(
    loc="upper left",
    fontsize=10,
    frameon=True
)

fig.suptitle(
    "Tabela 2 – porównanie wyników Axelroda z replikacją\n"
    "Siatka 10 × 10, po 10 powtórzeń",
    fontsize=14
)

fig.tight_layout(rect=[0, 0, 1, 0.91])

output_dir = Path("plots/table2")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "table2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
