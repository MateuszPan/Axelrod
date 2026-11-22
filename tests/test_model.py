import numpy as np
from model import (
    create_grid,
    get_neighbors,
    interaction_step,
    is_stable,
    run_simulation,
    count_regions,
    count_active_pairs,
    is_active_pair,
    update_active_pairs,
    run_simulation_optimized
)

def test_count_active_pairs():
    grid = np.array([
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]]
    ])

    assert count_active_pairs(grid) == 2
    assert is_stable(grid) == False

def test_active_pairs_consistency():
    for seed in range(100):
        grid = create_grid(n=5, F=5, q=3, seed=seed)

        active_pairs = count_active_pairs(grid)
        stable = is_stable(grid)

        assert (active_pairs == 0) == stable


def test_is_active_pair():
    grid = np.array([
        [[0, 1, 2], [0, 1, 0]],
        [[1, 2, 0], [0, 1, 2]]
    ])

    assert is_active_pair(grid, 0, 0, 0, 1)
    assert not is_active_pair(grid, 0, 0, 1, 0)
    assert not is_active_pair(grid, 0, 0, 1, 1)


def test_active_pairs_update():
    grid = np.array([
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]]
    ])

    active_pairs = count_active_pairs(grid)
    old_culture = grid[0, 0].copy()

    grid[0, 0, 1] = 1

    updated_pairs = update_active_pairs(
        grid, 0, 0, old_culture, active_pairs
    )

    assert active_pairs == 2
    assert updated_pairs == 2
    assert updated_pairs == count_active_pairs(grid)


def test_optimized_simulation():
    seed = 123

    rng_grid = np.random.default_rng(seed)
    initial_grid = create_grid(3, 5, 3, rng=rng_grid)

    grid_original = initial_grid.copy()
    grid_optimized = initial_grid.copy()

    rng_original = np.random.default_rng(seed + 1)
    rng_optimized = np.random.default_rng(seed + 1)

    result_original = run_simulation(
        grid_original,
        rng_original,
        max_steps=100000
    )

    result_optimized = run_simulation_optimized(
        grid_optimized,
        rng_optimized,
        max_steps=100000
    )

    assert np.array_equal(
        result_original["grid"],
        result_optimized["grid"]
    )

    assert result_original["steps"] == result_optimized["steps"]
    assert result_original["changes"] == result_optimized["changes"]

    assert is_stable(result_optimized["grid"])

    assert count_regions(
        result_original["grid"]
    ) == count_regions(result_optimized["grid"])


def test_optimized_multiple_seeds():
    for seed in range(20):
        rng_grid = np.random.default_rng(seed)
        initial_grid = create_grid(3, 5, 3, rng=rng_grid)

        grid_original = initial_grid.copy()
        grid_optimized = initial_grid.copy()

        rng_original = np.random.default_rng(seed + 1000)
        rng_optimized = np.random.default_rng(seed + 1000)

        result_original = run_simulation(
            grid_original,
            rng_original,
            max_steps=100000
        )

        result_optimized = run_simulation_optimized(
            grid_optimized,
            rng_optimized,
            max_steps=100000
        )

        assert np.array_equal(
            result_original["grid"],
            result_optimized["grid"]
        )

        assert result_original["steps"] == result_optimized["steps"]
        assert result_original["changes"] == result_optimized["changes"]

        assert is_stable(result_optimized["grid"])
        assert count_active_pairs(result_optimized["grid"]) == 0

        assert count_regions(
            result_original["grid"]
        ) == count_regions(result_optimized["grid"])


def test_grid_shape():
    grid = create_grid(n=3, F=5, q=3, seed=123)
    assert grid.shape == (3, 3, 5)


def test_values():
    grid = create_grid(n=3, F=5, q=3, seed=123)
    assert np.all((grid >= 0) & (grid < 3))


def test_reproducibility():
    grid1 = create_grid(n=3, F=5, q=3, seed=123)
    grid2 = create_grid(n=3, F=5, q=3, seed=123)
    assert np.array_equal(grid1, grid2)


def test_corner_neighbors():
    neighbors = get_neighbors(0, 0, 3)
    assert set(neighbors) == {(1, 0), (0, 1)}


def test_edge_neighbors():
    neighbors = get_neighbors(0, 1, 3)
    assert set(neighbors) == {(0, 0), (0, 2), (1, 1)}


def test_center_neighbors():
    neighbors = get_neighbors(1, 1, 3)
    assert set(neighbors) == {
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2)
    }


def test_interaction_step():
    rng = np.random.default_rng(123)
    grid = create_grid(n=3, F=5, q=3, seed=123)

    before = grid.copy()

    changed = interaction_step(grid, rng)

    differences = np.sum(before != grid)

    if changed:
        assert differences == 1
    else:
        assert differences == 0


def test_interaction_details():
    rng = np.random.default_rng(123)
    grid = create_grid(n=3, F=5, q=3, seed=123)

    before = grid.copy()

    details = interaction_step(grid, rng, return_details=True)

    assert isinstance(details, dict)
    assert 0 <= details["probability"] <= 1
    assert 0 <= details["common_features"] <= 5

    row = details["agent_row"] - 1
    col = details["agent_col"] - 1

    neighbor_row = details["neighbor_row"] - 1
    neighbor_col = details["neighbor_col"] - 1

    assert (neighbor_row, neighbor_col) in get_neighbors(row, col, 3)

    if details["changed"]:
        feature = details["feature"] - 1

        assert details["value_before"] == before[row, col, feature]
        assert details["value_after"] == grid[row, col, feature]
        assert details["value_after"] == before[
            neighbor_row, neighbor_col, feature
        ]

        assert np.sum(before != grid) == 1
    else:
        assert np.array_equal(before, grid)

def test_stability():
    grid = np.zeros((3, 3, 5), dtype=int)
    assert is_stable(grid) is True

    grid[0, 0] = [1, 0, 0, 0, 0]
    assert is_stable(grid) is False

    grid_single = np.array([
        [[0], [1]],
        [[1], [0]]
    ])
    assert is_stable(grid_single) is True


def test_run_simulation():
    rng = np.random.default_rng(123)

    grid = np.zeros((3, 3, 5), dtype=int)

    result = run_simulation(grid, rng)

    assert is_stable(result["grid"])
    assert result["steps"] == 0
    assert result["changes"] == 0

    assert np.array_equal(
        result["grid"], grid
    )


def test_count_regions():
    grid1 = np.zeros((3, 3, 2), dtype=int)
    assert count_regions(grid1) == 1

    grid2 = np.array([
        [[0, 0], [0, 0], [1, 1]],
        [[0, 0], [0, 0], [1, 1]],
        [[0, 0], [0, 0], [1, 1]]
    ])
    assert count_regions(grid2) == 2

    grid3 = np.array([
        [[0, 0], [1, 1]],
        [[1, 1], [0, 0]]
    ])
    assert count_regions(grid3) == 4


if __name__ == "__main__":
    test_grid_shape()
    test_values()
    test_reproducibility()
    test_corner_neighbors()
    test_edge_neighbors()
    test_center_neighbors()
    test_interaction_step()
    test_interaction_details()
    test_stability()
    test_run_simulation()
    test_count_regions()
    test_count_active_pairs()
    test_count_active_pairs()
    test_active_pairs_consistency()
    test_is_active_pair()
    test_active_pairs_update()
    test_optimized_simulation()
    test_optimized_multiple_seeds()
    print("Wszystkie 17 testów zakończone pomyślnie.")
    