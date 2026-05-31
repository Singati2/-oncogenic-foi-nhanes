"""NumPyro Bayesian implementation of the reverse-catalytic model.

Likelihood
----------
    n_pos[i] ~ Binomial(n_total[i], P(a_i)),
    P(a) = (lambda / (lambda+omega)) * (1 - exp(-(lambda+omega) * a))

Priors (weakly informative, both on the same scale as the constant-FOI fit)
--------------------------------------------------------------------------
- lambda ~ HalfNormal(0.01)
      Justification: the constant-FOI baseline returned lambda_hat ~ 0.0044
      on these data. HalfNormal(0.01) places ~95% of its mass below 0.02
      (about 5x lambda_hat), so it is weakly informative on the empirically
      reasonable scale and excludes physically implausible large values
      (e.g. lambda > 0.1 implies mean age of infection < 10 years).
- omega  ~ HalfNormal(0.01)
      Justification: published HPV IgG antibody waning timescales suggest a
      seroreversion rate per year on the order of 0.001-0.02 yr^-1, the same
      scale as lambda. HalfNormal(0.01) is the matching weakly-informative
      default; the data identify the ratio via the apparent plateau and rate
      of approach to it.

Deterministic transforms exposed in the posterior
-------------------------------------------------
- steady_state = lambda / (lambda + omega)  (asymptotic seroprevalence)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from numpyro.infer import MCMC, NUTS


def reverse_catalytic_model(
    ages: jnp.ndarray,
    n_total: jnp.ndarray,
    n_pos: jnp.ndarray | None = None,
    lambda_prior_scale: float = 0.01,
    omega_prior_scale: float = 0.01,
) -> None:
    """NumPyro reverse-catalytic with binomial observations.

    Prior scales are kwargs so prior-sensitivity sweeps can vary them
    without modifying the model. Defaults preserve the original spec.
    """
    lambda_ = numpyro.sample("lambda", dist.HalfNormal(lambda_prior_scale))
    omega = numpyro.sample("omega", dist.HalfNormal(omega_prior_scale))
    sum_ = lambda_ + omega
    numpyro.deterministic("steady_state", lambda_ / sum_)
    p = (lambda_ / sum_) * (1.0 - jnp.exp(-sum_ * ages))
    numpyro.sample(
        "obs", dist.Binomial(total_count=n_total, probs=p), obs=n_pos
    )


def fit_reverse_catalytic(
    ages: ArrayLike,
    n_total: ArrayLike,
    n_pos: ArrayLike,
    *,
    n_warmup: int = 500,
    n_samples: int = 1000,
    n_chains: int = 1,
    seed: int = 0,
    progress_bar: bool = False,
    lambda_prior_scale: float = 0.01,
    omega_prior_scale: float = 0.01,
) -> MCMC:
    """Run NUTS on the reverse-catalytic model. Returns the fitted MCMC."""
    kernel = NUTS(reverse_catalytic_model)
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
        lambda_prior_scale=float(lambda_prior_scale),
        omega_prior_scale=float(omega_prior_scale),
    )
    return mcmc
