"""Tests for the reverse-catalytic (seroreversion) model.

- Closed-form solver invariants and limits (fast)
- Steady-state helper (fast)
- Simulator validity (fast)
- NUTS recovery of lambda, omega (slow, marked)
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.model.bayesian_reverse import fit_reverse_catalytic
from analysis.model.catalytic import constant_foi_seroprevalence
from analysis.model.reverse_catalytic import (
    reverse_catalytic_seroprevalence,
    simulate_reverse_serology,
    steady_state,
)


# ---------- solver invariants ----------

def test_reverse_at_zero_age_is_zero() -> None:
    """P(0) = 0 regardless of (lambda, omega)."""
    for lam, om in [(0.005, 0.002), (0.05, 0.0), (0.0, 0.01), (0.01, 0.01)]:
        p = reverse_catalytic_seroprevalence(lam, om, np.array([0.0]))
        assert p[0] == 0.0


def test_reverse_recovers_constant_catalytic_when_omega_zero() -> None:
    """omega -> 0 reduces to P(a) = 1 - exp(-lambda * a)."""
    lam = 0.005
    ages = np.linspace(0, 80, 20)
    p_rev = reverse_catalytic_seroprevalence(lam, 0.0, ages)
    p_const = constant_foi_seroprevalence(lam, ages)
    np.testing.assert_allclose(p_rev, p_const, atol=1e-12)


def test_reverse_is_zero_when_lambda_zero() -> None:
    """lambda = 0 -> P(a) = 0 for all a (no force of infection)."""
    p = reverse_catalytic_seroprevalence(0.0, 0.01, np.linspace(0, 100, 50))
    assert (p == 0.0).all()


def test_reverse_is_monotone_non_decreasing() -> None:
    """P(a) is monotone non-decreasing in a (analytical property)."""
    ages = np.linspace(0, 100, 200)
    for lam, om in [(0.005, 0.001), (0.005, 0.01), (0.05, 0.05), (0.001, 0.05)]:
        p = reverse_catalytic_seroprevalence(lam, om, ages)
        assert (np.diff(p) >= -1e-15).all(), f"non-monotone at (λ={lam}, ω={om})"


def test_reverse_asymptotes_to_steady_state() -> None:
    """P(a) -> lambda / (lambda + omega) as a -> inf."""
    lam, om = 0.005, 0.002
    p_inf = reverse_catalytic_seroprevalence(lam, om, np.array([10_000.0]))
    expected_ss = lam / (lam + om)
    assert abs(p_inf[0] - expected_ss) < 1e-12
    assert abs(steady_state(lam, om) - expected_ss) < 1e-12


def test_steady_state_zero_when_lambda_and_omega_zero() -> None:
    assert steady_state(0.0, 0.0) == 0.0


# ---------- simulator ----------

def test_simulate_reverse_returns_valid_counts(rng_seed: int) -> None:
    ages = np.arange(10, 71, 10).astype(float)
    n_per = 200
    out = simulate_reverse_serology(0.005, 0.002, ages,
                                    n_per_age=n_per, seed=rng_seed)
    assert out.dtype.kind in ("i", "u")
    assert (out >= 0).all() and (out <= n_per).all()
    assert out.shape == ages.shape


# ---------- parameter recovery (slow) ----------

@pytest.mark.slow
def test_recovers_lambda_and_omega_within_ci(rng_seed: int) -> None:
    """NUTS recovers (lambda, omega) within 95% CrI on a clean simulated dataset."""
    true_lambda = 0.005
    true_omega = 0.002
    ages = np.arange(10, 81, 5).astype(float)   # 15 age points, broader range
    n_per_age = 400
    n_pos = simulate_reverse_serology(
        true_lambda, true_omega, ages, n_per_age=n_per_age, seed=rng_seed,
    )
    n_total = np.full_like(ages, n_per_age, dtype=int)

    mcmc = fit_reverse_catalytic(
        ages, n_total, n_pos,
        n_warmup=750, n_samples=1500,
        seed=rng_seed,
    )
    samp = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    lo_l, hi_l = np.quantile(samp["lambda"], [0.025, 0.975])
    lo_o, hi_o = np.quantile(samp["omega"], [0.025, 0.975])

    assert lo_l <= true_lambda <= hi_l, (
        f"true λ={true_lambda} outside 95% CrI [{lo_l:.5f}, {hi_l:.5f}]"
    )
    assert lo_o <= true_omega <= hi_o, (
        f"true ω={true_omega} outside 95% CrI [{lo_o:.5f}, {hi_o:.5f}]"
    )
