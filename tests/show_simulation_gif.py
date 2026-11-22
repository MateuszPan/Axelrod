import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

from model import create_grid, run_simulation


n = 3
F = 5
q = 3
seed = 123

rng_grid = np.random.default_rng(seed)
rng_interactions = np.random.default_rng(seed + 1)

grid = create_grid(n, F, q, rng=rng_grid)

result = run_simulation(
    grid, rng_interactions,
    max_steps=100000,
    save_history=True
)

history = result["history"]

fig, axes = plt.subplots(1, F, figsize=(15, 3.5))


def update(frame_index):
    state = history[frame_index]

    for feature in range(F):
        ax = axes[feature]
        ax.clear()

        ax.imshow(
            state[:, :, feature],
            cmap="viridis",
            vmin=0,
            vmax=q - 1
        )

        ax.set_title(f"Cecha {feature + 1}")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

        for row in range(n):
            for col in range(n):
                ax.text(
                    col, row,
                    str(state[row, col, feature]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=14,
                    fontweight="bold"
                )

    fig.suptitle(
        f"Model Axelroda - zmiana {frame_index} "
        f"z {result['changes']}"
    )

    return axes


animation = FuncAnimation(
    fig,
    update,
    frames=len(history),
    interval=200,
    repeat=False
)

output_dir = Path("tests/plots")
output_dir.mkdir(parents=True, exist_ok=True)

gif_path = output_dir / (
    f"simulation_n{n}_F{F}_q{q}_seed{seed}.gif"
)

animation.save(
    gif_path,
    writer=PillowWriter(fps=5),
    dpi=100
)

plt.close(fig)

print(f"Zapisano animacje: {gif_path}")
print(f"Liczba klatek: {len(history)}")
print(f"Liczba prob: {result['steps']}")
print(f"Liczba zmian: {result['changes']}")
