"""Constant force-of-infection (FOI) catalytic model: solver + simulator.

Mathematical model
------------------
Under a constant force of infection lambda, the susceptible fraction S(a)
satisfies the ODE

    dS(a)/da = -lambda * S(a),   S(0) = 1,

which has the closed-form solution

    S(a) = exp(-lambda * a),

so the seroprevalence at age a is

    P(a) = 1 - exp(-lambda * a).

This module provides the analytical solver and a binomial-data simulator used
for parameter-recovery tests of the Bayesian implementation in `bayesian.py`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def constant_foi_seroprevalence(lambda_: float, ages: ArrayLike) -> np.ndarray:
    """Analytical seroprevalence under constant FOI.

    Returns P(a) = 1 - exp(-lambda * a) elementwise for each age in `ages`.
    """
    return 1.0 - np.exp(-float(lambda_) * np.asarray(ages, dtype=float))


def simulate_serology(
    lambda_: float,
    ages: ArrayLike,
    n_per_age: int | ArrayLike,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Simulate observed seropositive counts per age under constant FOI.

    For each age a in `ages`, draws n_pos ~ Binomial(n_per_age, P(a)) where
    P(a) = 1 - exp(-lambda * a). Returns an integer array shaped like `ages`.

    Parameters
    ----------
    lambda_     : force of infection (per year)
    ages        : age values (years)
    n_per_age   : number of individuals tested at each age (scalar or array)
    seed        : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    ages_arr = np.asarray(ages, dtype=float)
    p = constant_foi_seroprevalence(lambda_, ages_arr)
    n_per_age_arr = np.broadcast_to(
        np.asarray(n_per_age, dtype=np.int64), ages_arr.shape
    )
    return rng.binomial(n_per_age_arr, p)
