"""Unit tests for the constant-FOI catalytic solver and simulator.

Reference: under constant force of infection λ, the catalytic model
    dS/da = -λ · S(a),  S(0) = 1
has the analytical solution  S(a) = exp(-λ · a),  so
    P(a) = 1 - exp(-λ · a).
Any numerical implementation must match this exactly.
"""

from __future__ import annotations

import numpy as np

from analysis.model.catalytic import constant_foi_seroprevalence, simulate_serology


def test_constant_foi_matches_analytical() -> None:
    """Solver must equal 1 - exp(-λa) for the closed-form constant-FOI case."""
    lambda_ = 0.05
    ages = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 60.0])
    expected = 1.0 - np.exp(-lambda_ * ages)
    got = constant_foi_seroprevalence(lambda_, ages)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_seroprevalence_zero_at_birth() -> None:
    """P(0) = 0 regardless of λ."""
    for lambda_ in (0.01, 0.05, 0.10, 0.50):
        p = constant_foi_seroprevalence(lambda_, np.array([0.0]))
        assert p[0] == 0.0


def test_seroprevalence_monotone_nondecreasing() -> None:
    """P(a) is monotone non-decreasing in a for any λ > 0."""
    ages = np.linspace(0, 80, 100)
    for lambda_ in (0.01, 0.05, 0.10):
        p = constant_foi_seroprevalence(lambda_, ages)
        assert np.all(np.diff(p) >= -1e-15), f"non-monotone for λ={lambda_}"


def test_seroprevalence_approaches_one_at_high_age() -> None:
    """For positive λ, P(a) → 1 as a → ∞."""
    p = constant_foi_seroprevalence(0.10, np.array([1000.0]))
    assert p[0] > 0.999


def test_simulate_serology_returns_valid_counts(rng_seed: int) -> None:
    """Simulated counts are nonneg integers ≤ n_per_age, shaped like ages."""
    ages = np.arange(10, 71, 10).astype(float)
    n_per_age = 100
    n = simulate_serology(0.05, ages, n_per_age=n_per_age, seed=rng_seed)
    assert n.dtype.kind in ("i", "u")
    assert (n >= 0).all()
    assert (n <= n_per_age).all()
    assert n.shape == ages.shape


def test_simulate_serology_seed_is_reproducible() -> None:
    """Same seed -> identical simulated counts."""
    ages = np.arange(10, 71, 5).astype(float)
    a = simulate_serology(0.05, ages, n_per_age=100, seed=42)
    b = simulate_serology(0.05, ages, n_per_age=100, seed=42)
    np.testing.assert_array_equal(a, b)
