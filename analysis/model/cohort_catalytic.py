"""Cohort-structured catalytic model: per-cohort constant FOI.

Within each birth cohort k, the model is the standard constant-FOI catalytic

    P(a | k) = 1 - exp(-lambda_k * a),

i.e. each cohort gets its own age-independent force of infection lambda_k.
Cross-cohort variation in lambda captures, in a tractable way, the cohort
effects that the single-FOI baseline cannot represent.

This module provides the analytical solver and a binomial simulator used by
parameter-recovery tests of the Bayesian implementation in `bayesian_cohort`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def cohort_seroprevalence(
    lambdas: ArrayLike,
    ages: ArrayLike,
    cohort_idx: ArrayLike,
) -> np.ndarray:
    """Per-observation seroprevalence P_i = 1 - exp(-lambdas[cohort_idx[i]] * ages[i])."""
    lambdas_arr = np.asarray(lambdas, dtype=float)
    ages_arr = np.asarray(ages, dtype=float)
    idx = np.asarray(cohort_idx, dtype=int)
    lams_per_obs = lambdas_arr[idx]
    return 1.0 - np.exp(-lams_per_obs * ages_arr)


def simulate_cohort_serology(
    lambdas: ArrayLike,
    ages: ArrayLike,
    cohort_idx: ArrayLike,
    n_per_obs: int | ArrayLike,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Simulate seropositive counts per (age, cohort) cell."""
    rng = np.random.default_rng(seed)
    ages_arr = np.asarray(ages, dtype=float)
    p = cohort_seroprevalence(lambdas, ages_arr, cohort_idx)
    n_arr = np.broadcast_to(np.asarray(n_per_obs, dtype=np.int64), ages_arr.shape)
    return rng.binomial(n_arr, p)
