"""Fast smoke test for the Bayesian catalytic fit pipeline.

Runs a tiny NUTS chain (~5 s including JAX warmup) to verify the model code
path is wired correctly end-to-end. Catches API/shape errors that would
otherwise only surface in the slow recovery test.

Intentionally NOT marked slow — runs in CI on every push.
"""

from __future__ import annotations

import numpy as np

from analysis.model.bayesian import fit_catalytic
from analysis.model.catalytic import simulate_serology


def test_fit_runs_and_returns_sensible_posterior(rng_seed: int) -> None:
    """Minimal fit completes; posterior shape correct and λ samples sensible."""
    ages = np.arange(20, 51, 10).astype(float)   # 4 age points
    n_per_age = 100
    n_pos = simulate_serology(0.05, ages, n_per_age, seed=rng_seed)
    n_total = np.full_like(ages, n_per_age, dtype=int)

    mcmc = fit_catalytic(
        ages, n_total, n_pos,
        n_warmup=50, n_samples=100,
        seed=rng_seed,
    )
    samples = np.asarray(mcmc.get_samples()["lambda"])

    assert samples.shape == (100,)
    assert (samples > 0).all(), "λ samples must be positive (HalfNormal prior)"
    assert samples.mean() < 1.0, "posterior mean should be O(0.01-0.1), not blown up"
    assert np.isfinite(samples).all(), "no NaN / inf in posterior samples"
