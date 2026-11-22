"""Unit tests for the error-driven sample allocation and neighbour generation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aircraft_heat import config                                  # noqa: E402
from aircraft_heat.adapt import allocate, neighbours               # noqa: E402


def test_allocation_scales_with_error():
    n = allocate([1.0, 1.99, 2.0, 5.0, 10.0, 20.0, 100.0], N=3)
    assert n.tolist() == [0, 0, 1, 2, 3, 6, 12]          # none below 2 K; N at 10 K; 2N at 20 K; capped at 4N


def test_neighbours_are_valid_and_close():
    rng = np.random.default_rng(0)
    v = config.sample_conditions(1, rng)[0]
    nb = neighbours(v, 8, rng)
    assert nb.shape == (8, config.N_PARAMS)
    M = config.mach_number(nb[:, 0], nb[:, 1])
    assert ((M >= config.MACH_MIN) & (M <= config.MACH_MAX)).all()
    assert (np.abs(config.to_unit(nb) - config.to_unit(v)) < 0.5).all()


if __name__ == "__main__":
    test_allocation_scales_with_error(); test_neighbours_are_valid_and_close(); print("adapt tests passed")
