import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

from model import create_grid, get_neighbors, run_simulation, count_regions


n = 3
F = 5
q = 3
seed = 123

rng_grid = np.random.default_rng(seed)
rng_interactions = np.random.default_rng(seed + 1)

grid = create_grid(n, F, q, rng=rng_grid)

result = run_simulation(grid, rng_interactions)
final_grid = result["grid"]


def identify_regions(grid):
    n = grid.shape[0]
    region_map = np.zeros((n, n), dtype=int)
    visited = set()
    region_id = 0

    for row in range(n):
        for col in range(n):
            if (row, col) in visited:
                continue

            region_id += 1
            culture = grid[row, col].copy()
            stack = [(row, col)]
            visited.add((row, col))

            while stack:
                current_row, current_col = stack.pop()
                region_map[current_row, current_col] = region_id

                for nr, nc in get_neighbors(current_row, current_col, n):
                    if (nr, nc) in visited:
                        continue

                    if np.array_equal(grid[nr, nc], culture):
                        visited.add((nr, nc))
                        stack.append((nr, nc))

    return region_map


region_map = identify_regions(final_grid)
regions = count_regions(final_grid)

assert len(np.unique(region_map)) == regions

fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(region_map, cmap="tab20", vmin=1, vmax=max(regions, 2))

for row in range(n):
    for col in range(n):
        ax.text(
            col, row,
            str(region_map[row, col]),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="black"
        )

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

ax.set_title(f"Stabilne regiony kulturowe: {regions}")

output_dir = Path("tests/plots")
output_dir.mkdir(parents=True, exist_ok=True)

filename = output_dir / f"regions_n{n}_F{F}_q{q}_seed{seed}.png"

plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.show()

print("Mapa regionów:")
print(region_map)
print("Liczba stabilnych regionów:", regions)
print("Wykres zapisano:", filename)

test_grid = np.array([
    [[0, 0], [1, 1], [0, 0]],
    [[1, 1], [1, 1], [1, 1]],
    [[0, 0], [1, 1], [0, 0]]
])

test_map = identify_regions(test_grid)
test_regions = count_regions(test_grid)

print("\nMapa regionów dla siatki testowej:")
print(test_map)
print("Liczba regionów:", test_regions)

assert test_regions == 5
assert len(np.unique(test_map)) == 5
