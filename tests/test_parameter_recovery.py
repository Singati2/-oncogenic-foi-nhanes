"""Parameter-recovery tests (project plan §8).

For known synthetic λ values, the Bayesian fit should recover λ within the 95%
credible interval at least 95% of the time across simulated datasets.

This is the central test that the TDD plan asks to be green before any real-data
fit is performed.
"""

import pytest


@pytest.mark.slow
@pytest.mark.skip(reason="inference not yet implemented — see analysis/inference/")
def test_recovers_constant_foi_within_ci(rng_seed: int) -> None:
    """Recovery coverage ≥ 95% for constant λ across 100 sim datasets."""
    # true_lambda = 0.05
    # n_sims = 100
    # n_per_sim = 1_000
    #
    # covered = 0
    # for s in range(n_sims):
    #     data = simulate_serology(lambda_=true_lambda, n=n_per_sim, seed=rng_seed + s)
    #     posterior = fit_catalytic(data)
    #     lo, hi = posterior["lambda"].quantile([0.025, 0.975])
    #     if lo <= true_lambda <= hi:
    #         covered += 1
    #
    # assert covered / n_sims >= 0.95

    raise NotImplementedError
