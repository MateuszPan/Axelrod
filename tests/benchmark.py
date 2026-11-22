import time
import numpy as np

from model import (
    create_grid,
    run_simulation,
    run_simulation_optimized
)

n = 10
F = 5
q = 3
seed = 123

rng_grid = np.random.default_rng(seed)
initial_grid = create_grid(n, F, q, rng=rng_grid)

grid_original = initial_grid.copy()
grid_optimized = initial_grid.copy()

rng_original = np.random.default_rng(seed + 1)
rng_optimized = np.random.default_rng(seed + 1)

start = time.perf_counter()

result_original = run_simulation(
    grid_original,
    rng_original,
    max_steps=1000000
)

time_original = time.perf_counter() - start

start = time.perf_counter()

result_optimized = run_simulation_optimized(
    grid_optimized,
    rng_optimized,
    max_steps=1000000
)

time_optimized = time.perf_counter() - start

print("Wersja pierwotna:")
print("Czas:", round(time_original, 4), "s")
print("Liczba prób:", result_original["steps"])
print("Liczba zmian:", result_original["changes"])

print("\nWersja zoptymalizowana:")
print("Czas:", round(time_optimized, 4), "s")
print("Liczba prób:", result_optimized["steps"])
print("Liczba zmian:", result_optimized["changes"])

print("\nPrzyspieszenie:", round(time_original / time_optimized, 2), "razy")

assert np.array_equal(result_original["grid"], result_optimized["grid"])
assert result_original["steps"] == result_optimized["steps"]
assert result_original["changes"] == result_optimized["changes"]