import os
import numpy as np
import matplotlib.pyplot as plt

from model import create_grid

n = 3
F = 5
q = 3
seed = 123

grid = create_grid(n, F, q, seed)

fig, axes = plt.subplots(1, F, figsize=(15, 3.5))

for feature in range(F):
    ax = axes[feature]

    ax.imshow(
        grid[:, :, feature],
        cmap="viridis",
        vmin=0,
        vmax=q - 1
    )

    ax.set_title(f"Cecha {feature + 1}")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(n):
        for col in range(n):
            value = grid[row, col, feature]
            ax.text(
                col, row, str(value),
                ha="center",
                va="center",
                color="white",
                fontsize=14,
                fontweight="bold"
            )

plt.suptitle("Model Axelroda - poczatkowe cechy agentow")
plt.tight_layout()

os.makedirs("tests/plots", exist_ok=True)
filename = f"tests/plots/grid_n{n}_F{F}_q{q}_seed{seed}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")

plt.show()

print("Poczatkowa siatka agentow:")
print(grid)
print(f"Wykres zapisano do pliku: {filename}")