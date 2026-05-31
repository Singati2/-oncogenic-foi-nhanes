"""Reverse-catalytic (constant FOI with seroreversion) model: solver + simulator.

Mathematical model
------------------
Constant force of infection lambda and constant seroreversion rate omega:

    dS/da = -lambda * S(a) + omega * (1 - S(a)),    S(0) = 1.

This is a linear first-order ODE; rewriting as dS/da = omega - (lambda+omega) S
gives the closed-form solution

    S(a) = omega / (lambda+omega) + (lambda/(lambda+omega)) * exp(-(lambda+omega)*a),

so seroprevalence

    P(a) = 1 - S(a) = (lambda / (lambda+omega)) * (1 - exp(-(lambda+omega)*a))

with steady state P(inf) = lambda / (lambda+omega).

Important shape constraint
--------------------------
For lambda >= 0 and omega >= 0, P(a) is **monotone non-decreasing** in a. The
model can produce a plateau (approach to steady state) but **cannot produce a
declining prevalence at older ages**. An observed decline therefore cannot be
attributed to seroreversion alone within this model class.

Limits
------
- omega -> 0: P(a) -> 1 - exp(-lambda*a)  (recovers constant-FOI catalytic)
- lambda -> 0: P(a) -> 0                  (no infection)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def reverse_catalytic_seroprevalence(
    lambda_: float, omega: float, ages: ArrayLike
) -> np.ndarray:
    """Closed-form P(a) for reverse-catalytic with constant lambda and omega."""
    lam = float(lambda_)
    om = float(omega)
    a = np.asarray(ages, dtype=float)
    sum_ = lam + om
    if sum_ <= 0:
        return np.zeros_like(a)
    return (lam / sum_) * (1.0 - np.exp(-sum_ * a))


def steady_state(lambda_: float, omega: float) -> float:
    """Asymptotic seroprevalence P(inf) = lambda / (lambda+omega)."""
    sum_ = float(lambda_) + float(omega)
    return float(lambda_) / sum_ if sum_ > 0 else 0.0


def simulate_reverse_serology(
    lambda_: float,
    omega: float,
    ages: ArrayLike,
    n_per_age: int | ArrayLike,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Simulate seropositive counts per age under reverse-catalytic dynamics."""
    rng = np.random.default_rng(seed)
    ages_arr = np.asarray(ages, dtype=float)
    p = reverse_catalytic_seroprevalence(lambda_, omega, ages_arr)
    n_arr = np.broadcast_to(
        np.asarray(n_per_age, dtype=np.int64), ages_arr.shape
    )
    return rng.binomial(n_arr, p)
