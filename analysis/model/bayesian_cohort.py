"""NumPyro hierarchical cohort-structured catalytic model.

Likelihood
----------
    n_pos[i] ~ Binomial(n_total[i], P(a_i | k_i)),
    P(a | k) = 1 - exp(-exp(log_lambda[k]) * a)

Prior (weakly hierarchical)
---------------------------
    mu          ~ Normal(log(0.005), 1.0)
    sigma       ~ HalfNormal(1.0)
    log_lambda  ~ Normal(mu, sigma)        (one per cohort)

Notes
-----
Centered parameterization is fine here because the cohort count is small
(typically 3-5) and the per-cohort data are well-identified; we have not
observed funnel pathologies during recovery testing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from numpyro.infer import MCMC, NUTS


def cohort_catalytic_model(
    ages: jnp.ndarray,
    cohort_idx: jnp.ndarray,
    n_cohorts: int,
    n_total: jnp.ndarray,
    n_pos: jnp.ndarray | None = None,
) -> None:
    """Hierarchical cohort-structured catalytic with binomial observations."""
    mu = numpyro.sample("mu", dist.Normal(jnp.log(0.005), 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    with numpyro.plate("cohorts", n_cohorts):
        log_lambda = numpyro.sample("log_lambda", dist.Normal(mu, sigma))

    # Deterministic per-cohort lambdas, exposed in the posterior for downstream use.
    numpyro.deterministic("lambda", jnp.exp(log_lambda))

    lambda_per_obs = jnp.exp(log_lambda)[cohort_idx]
    p = 1.0 - jnp.exp(-lambda_per_obs * ages)
    numpyro.sample(
        "obs", dist.Binomial(total_count=n_total, probs=p), obs=n_pos
    )


def fit_cohort_catalytic(
    ages: ArrayLike,
    cohort_idx: ArrayLike,
    n_cohorts: int,
    n_total: ArrayLike,
    n_pos: ArrayLike,
    *,
    n_warmup: int = 500,
    n_samples: int = 1000,
    n_chains: int = 1,
    seed: int = 0,
    progress_bar: bool = False,
) -> MCMC:
    """Run NUTS on the hierarchical cohort-structured catalytic model."""
    kernel = NUTS(cohort_catalytic_model)
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        ages=jnp.asarray(ages, dtype=jnp.float32),
        cohort_idx=jnp.asarray(cohort_idx, dtype=jnp.int32),
        n_cohorts=n_cohorts,
        n_total=jnp.asarray(n_total, dtype=jnp.int32),
        n_pos=jnp.asarray(n_pos, dtype=jnp.int32),
    )
    return mcmc
