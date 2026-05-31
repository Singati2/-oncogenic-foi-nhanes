"""Tests for the cohort-structured catalytic model.

- Solver invariants (fast)
- Simulator validity (fast)
- Birth-year and cohort assignment helpers (fast)
- Per-cohort parameter recovery via NUTS (slow, marked)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.inference.fit_hpv_cohort import (
    COHORT_BOUNDS,
    COHORT_LABELS,
    CYCLE_MIDPOINT,
    prepare,
)
from analysis.model.bayesian_cohort import fit_cohort_catalytic
from analysis.model.cohort_catalytic import (
    cohort_seroprevalence,
    simulate_cohort_serology,
)


# ---------- solver ----------

def test_cohort_seroprevalence_matches_per_cohort_closed_form() -> None:
    lambdas = np.array([0.02, 0.05])
    ages = np.array([10.0, 20.0, 30.0, 40.0])
    cohort_idx = np.array([0, 0, 1, 1])
    expected = 1.0 - np.exp(-lambdas[cohort_idx] * ages)
    got = cohort_seroprevalence(lambdas, ages, cohort_idx)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_cohort_seroprevalence_zero_lambda_gives_zero() -> None:
    lambdas = np.array([0.0, 0.05])
    ages = np.array([10.0, 50.0])
    cohort_idx = np.array([0, 0])
    out = cohort_seroprevalence(lambdas, ages, cohort_idx)
    assert (out == 0.0).all()


# ---------- simulator ----------

def test_simulate_cohort_returns_valid_counts(rng_seed: int) -> None:
    lambdas = np.array([0.02, 0.05, 0.10])
    ages = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    cohort_idx = np.array([0, 1, 2, 0, 1, 2])
    n_per = 100
    out = simulate_cohort_serology(lambdas, ages, cohort_idx, n_per_obs=n_per, seed=rng_seed)
    assert out.shape == ages.shape
    assert out.dtype.kind in ("i", "u")
    assert (out >= 0).all() and (out <= n_per).all()


# ---------- data prep helpers ----------

def test_cycle_midpoint_dict_complete() -> None:
    assert set(CYCLE_MIDPOINT.keys()) == {"C", "D", "E", "F"}
    assert CYCLE_MIDPOINT["C"] == 2004
    assert CYCLE_MIDPOINT["F"] == 2010


def test_cohort_labels_and_bounds_consistent() -> None:
    assert len(COHORT_LABELS) == 4
    # bounds = [-inf, 1955, 1970, 1985, inf] -> 4 intervals -> 4 labels
    assert len(COHORT_BOUNDS) == len(COHORT_LABELS) + 1


def test_prepare_derives_birth_year_and_cohort() -> None:
    df = pd.DataFrame(
        {
            "cycle": ["C", "F", "C", "D"],
            "age": [50, 25, 30, 18],
            "sero_hpv_hr": pd.array([1, 0, pd.NA, 1], dtype="Int8"),
        }
    )
    out = prepare(df, age_min=14, age_max=59)
    # NA dropped
    assert len(out) == 3
    # birth_year = cycle_midpoint - age
    expected_by = [2004 - 50, 2010 - 25, 2006 - 18]
    assert sorted(out["birth_year"].tolist()) == sorted(expected_by)
    # cohort labels assigned
    assert set(out["cohort"].astype(str)) <= set(COHORT_LABELS)


def test_prepare_drops_ages_out_of_range() -> None:
    df = pd.DataFrame(
        {
            "cycle": ["C", "C", "C"],
            "age": [10, 30, 70],
            "sero_hpv_hr": pd.array([1, 0, 1], dtype="Int8"),
        }
    )
    out = prepare(df, age_min=14, age_max=59)
    assert out["age"].tolist() == [30]


# ---------- parameter recovery (slow) ----------

@pytest.mark.slow
def test_cohort_fit_recovers_per_cohort_lambdas(rng_seed: int) -> None:
    """NUTS recovers each cohort's true lambda within 95% CrI on a clean sim."""
    true_lambdas = np.array([0.003, 0.005, 0.008])  # 3 cohorts
    n_cohorts = len(true_lambdas)
    ages = np.tile(np.arange(20, 61, 5, dtype=float), n_cohorts)
    cohort_idx = np.repeat(np.arange(n_cohorts), 9)
    n_per_cell = 300
    n_pos = simulate_cohort_serology(
        true_lambdas, ages, cohort_idx, n_per_obs=n_per_cell, seed=rng_seed
    )
    n_total = np.full_like(ages, n_per_cell, dtype=int)

    mcmc = fit_cohort_catalytic(
        ages=ages, cohort_idx=cohort_idx, n_cohorts=n_cohorts,
        n_total=n_total, n_pos=n_pos,
        n_warmup=500, n_samples=1000, seed=rng_seed,
    )
    samples = np.asarray(mcmc.get_samples()["lambda"])  # (S, K)

    for k, true_lam in enumerate(true_lambdas):
        lo, hi = np.quantile(samples[:, k], [0.025, 0.975])
        assert lo <= true_lam <= hi, (
            f"cohort {k}: true λ={true_lam} outside 95% CrI [{lo:.5f}, {hi:.5f}]"
        )
