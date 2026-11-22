import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from model import create_grid, interaction_step


n = 3
F = 5
q = 3
seed = 123

rng = np.random.default_rng(seed)
grid = create_grid(n, F, q, seed)

before = grid.copy()

attempts = 0
max_attempts = 10000

while attempts < max_attempts:
    attempts += 1

    details = interaction_step(grid, rng, return_details=True)

    if details["changed"]:
        break
else:
    raise RuntimeError("Nie wystapila zmiana w limicie prob.")

after = grid.copy()

changed_positions = np.argwhere(before != after)
row, col, feature = changed_positions[0]

fig, axes = plt.subplots(2, F, figsize=(15, 7))

for state_index, state in enumerate([before, after]):
    for feature_index in range(F):
        ax = axes[state_index, feature_index]

        ax.imshow(
            state[:, :, feature_index],
            cmap="viridis",
            vmin=0,
            vmax=q - 1
        )

        ax.set_title(
            f"{'Przed' if state_index == 0 else 'Po'} - "
            f"cecha {feature_index + 1}"
        )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for r in range(n):
            for c in range(n):
                ax.text(
                    c, r, str(state[r, c, feature_index]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=13,
                    fontweight="bold"
                )

        if feature_index == feature:
            ax.add_patch(
                plt.Rectangle(
                    (col - 0.5, row - 0.5),
                    1, 1,
                    fill=False,
                    edgecolor="red",
                    linewidth=4
                )
            )

plt.suptitle("Model Axelroda - pierwsza zmiana kulturowa")
plt.tight_layout()

output_dir = Path("tests/plots")
output_dir.mkdir(parents=True, exist_ok=True)

filename = output_dir / (
    f"interaction_n{n}_F{F}_q{q}_seed{seed}.png"
)

plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.show()

print(f"Liczba prob interakcji: {attempts}")
print(f"Zmieniony agent: wiersz {row + 1}, kolumna {col + 1}")
print(f"Zmieniona cecha: {feature + 1}")
print(f"Poprzednia wartosc: {before[row, col, feature]}")
print(f"Nowa wartosc: {after[row, col, feature]}")
print(f"Wykres zapisano: {filename}")

data_dir = Path("tests/data")

result = {
    "seed": seed,
    "n": n,
    "F": F,
    "q": q,
    "attempts": attempts,
    "agent_row": details["agent_row"],
    "agent_col": details["agent_col"],
    "neighbor_row": details["neighbor_row"],
    "neighbor_col": details["neighbor_col"],
    "common_features": details["common_features"],
    "probability": details["probability"],
    "changed": details["changed"],
    "feature": details["feature"],
    "value_before": details["value_before"],
    "value_after": details["value_after"]
}

df = pd.DataFrame([result])

csv_path = data_dir / f"interaction_n{n}_F{F}_q{q}_seed{seed}.csv"

df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"Dane zapisano do pliku: {csv_path}")


