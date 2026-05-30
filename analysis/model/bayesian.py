"""NumPyro Bayesian implementation of the constant-FOI catalytic model.

Likelihood:  n_pos[i] ~ Binomial(n_total[i], P(a_i))   with P(a) = 1 - exp(-lambda * a)
Prior:       lambda  ~ HalfNormal(scale=0.1)

The HalfNormal(0.1) prior is weakly informative for an oncogenic infection: it
places ~95% of its mass below lambda = 0.196 yr^-1 (median age of infection
about 3.5 years and above), comfortably covering the empirically plausible
range for HCV, HSV-2, EBV, *H. pylori*, and HPV-16/18 in the US population.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from numpyro.infer import MCMC, NUTS


def catalytic_model(
    ages: jnp.ndarray,
    n_total: jnp.ndarray,
    n_pos: jnp.ndarray | None = None,
) -> None:
    """NumPyro model for constant-FOI catalytic with binomial observations."""
    lambda_ = numpyro.sample("lambda", dist.HalfNormal(0.1))
    p = 1.0 - jnp.exp(-lambda_ * ages)
    numpyro.sample(
        "obs", dist.Binomial(total_count=n_total, probs=p), obs=n_pos
    )


def fit_catalytic(
    ages: ArrayLike,
    n_total: ArrayLike,
    n_pos: ArrayLike,
    *,
    n_warmup: int = 500,
    n_samples: int = 1000,
    n_chains: int = 1,
    seed: int = 0,
    progress_bar: bool = False,
) -> MCMC:
    """Run NUTS on the constant-FOI catalytic model. Returns the fitted MCMC."""
    kernel = NUTS(catalytic_model)
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
        n_total=jnp.asarray(n_total, dtype=jnp.int32),
        n_pos=jnp.asarray(n_pos, dtype=jnp.int32),
    )
    return mcmc
