# oncogenic-foi-nhanes

Multi-pathogen serocatalytic modeling of oncogenic infections in NHANES with neural-ODE force-of-infection estimation and projected cancer burden.

**Status:** scaffold — pre-data, pre-model. Repo seeded; analysis pipeline under construction.

## Project at a glance

Fit Bayesian hierarchical serocatalytic models to NHANES age-stratified serology for the major oncogenic pathogens (HPV, HCV, HSV-1, HSV-2, EBV, *H. pylori*), estimate age- and cohort-dependent force of infection $\lambda(a, c)$, extend with a neural-ODE component to relax functional-form assumptions, and project infection-attributable cancer burden under counterfactual scenarios (vaccination, screening, WHO elimination targets). A Nepal arm replicates the framework for HIC↔LMIC comparison.

## Why this project exists

Three concrete gaps in the 2024–2026 literature:

1. **No multi-pathogen serocatalytic model** has been fit to the NHANES oncogenic panel (the closest prior work is HPV-only and 6–8 years old).
2. **Co-infection / interaction structure** is biologically plausible but ~79% of co-infection models use non-empirical interaction parameters.
3. **Force of infection $\lambda(a,t)$** is forced into rigid parametric forms (piecewise constant, exponential, spline). A neural-ODE component lifts this constraint while preserving mechanistic priors.

## Data foundation

Primary analysis window: **NHANES 2003–2010** — the only continuous-cycle window where the five oncogenic pathogens of interest overlap in the same individuals. See [`docs/decisions/0001-window-selection.md`](docs/decisions/0001-window-selection.md) for the access-map analysis that drove this scope.

## Repository layout

```
analysis/
  data_pull/      NHANES + GLOBOCAN download / harmonization
  model/          Stan + NumPyro serocatalytic implementations
  neural_ode/     diffrax NeuralODE for λ(a, t)
  inference/      NUTS, VI, SBC runners
  scenarios/      counterfactual scenario simulator
  burden/         attributable-fraction cancer projection
  diagnostics/    R̂, ESS, posterior / prior predictive checks
  figures/        publication-quality outputs
  app/            Streamlit policy dashboard
tests/            pytest — unit, recovery, SBC, identifiability
docs/
  decisions/      Architecture Decision Records (ADRs)
.github/workflows ci.yml + nightly SBC suite
```

## Quick start (once the pipeline lands)

```bash
make install       # install deps via uv / renv
make pull          # download NHANES 2003-2010 files
make fit           # fit single-pathogen catalytic model (HPV) as smoke test
make test          # run unit + recovery suite
```

## Reproducibility commitment

- Containerized (Docker) — exact-version dependencies
- CI runs lint + fast tests on every push
- SBC + sampling smoke tests run nightly
- Every figure, table, and number traces back to a script + commit hash
- All randomness uses fixed seeds; CI verifies bit-reproducibility

## License

Code: [MIT](LICENSE). Derived documentation and data: CC-BY-4.0 (added at first data release).

## Citation

This project will mint a Zenodo DOI on first tagged release. Until then, cite the repository URL and commit hash.
