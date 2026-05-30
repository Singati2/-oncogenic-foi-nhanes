"""Parameter-recovery test for the constant-FOI Bayesian catalytic model.

A single NUTS fit on a clean simulated dataset must recover the true λ
within the 95% credible interval. This is the core sanity check that the
Bayesian implementation is correct.

Marked @pytest.mark.slow because it runs JAX compilation + NUTS sampling
(~10-30 s). Run explicitly with:  pytest -m slow
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.model.bayesian import fit_catalytic
from analysis.model.catalytic import simulate_serology


@pytest.mark.slow
def test_recovers_constant_foi_within_ci(rng_seed: int) -> None:
    """A single NUTS fit recovers true λ within the 95% CrI on a clean sim."""
    true_lambda = 0.05
    ages = np.arange(10, 71, 5).astype(float)   # 13 age points: 10, 15, ..., 70
    n_per_age = 200
    n_pos = simulate_serology(true_lambda, ages, n_per_age, seed=rng_seed)
    n_total = np.full_like(ages, n_per_age, dtype=int)

    mcmc = fit_catalytic(
        ages, n_total, n_pos,
        n_warmup=500, n_samples=1000,
        seed=rng_seed,
    )
    samples = np.asarray(mcmc.get_samples()["lambda"])
    lo, hi = np.quantile(samples, [0.025, 0.975])

    assert lo <= true_lambda <= hi, (
        f"true λ={true_lambda} not inside 95% CrI [{lo:.5f}, {hi:.5f}]"
    )
