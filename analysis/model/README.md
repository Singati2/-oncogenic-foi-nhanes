# `analysis/model/` — serocatalytic models

First piece of real science. Begins with the constant-FOI single-pathogen baseline; future modules will add cohort effects, multi-pathogen joint structure, and the neural-ODE extension per the project plan.

## What's here

| File | Purpose |
|---|---|
| `catalytic.py` | Analytical solver `P(a) = 1 - exp(-λa)` and a binomial-data simulator used by recovery tests |
| `bayesian.py` | NumPyro implementation: `catalytic_model` + `fit_catalytic` (NUTS) |

## Mathematical model (v0)

Under a **constant force of infection** λ:

```
dS/da = -λ · S(a),    S(0) = 1
=>  P(a) = 1 - exp(-λ · a)
```

Likelihood at age a_i with n_total_i individuals tested:

```
n_pos_i  ~  Binomial(n_total_i, P(a_i))
```

Prior:

```
λ  ~  HalfNormal(0.1)
```

## Tested invariants (see `tests/test_catalytic_solver.py`)

- Analytical solver matches `1 - exp(-λa)` exactly (atol 1e-12)
- P(0) = 0 regardless of λ
- P(a) is monotone non-decreasing in a for any λ > 0
- P(a) → 1 as a → ∞ for any λ > 0
- `simulate_serology` returns nonneg integer counts ≤ n_per_age

## Parameter recovery (see `tests/test_parameter_recovery.py`)

A single NUTS fit on a clean synthetic dataset (λ=0.05, ages 10–70, 200/age) recovers the true λ within the 95% credible interval. Marked `@pytest.mark.slow` because it runs JAX compilation + NUTS — typically 10–30 seconds. Run with `pytest -m slow`.

## What this baseline does NOT do (yet)

- Cohort effects (λ varying by birth year)
- Time-varying λ within a cohort
- Antibody waning / seroreversion
- Imperfect test sensitivity/specificity
- Multi-pathogen joint structure
- Neural-ODE flexibility for λ(a, t)

These are deliberate next steps once the baseline is rock-solid.
