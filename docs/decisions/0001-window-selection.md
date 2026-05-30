# ADR 0001 — Primary analysis window: NHANES 2003–2010

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** project lead

## Context

The original project plan assumed a joint 5-pathogen serocatalytic model with cohort effects could be fit across all continuous NHANES cycles. An access-map audit of CDC NHANES laboratory data (artifact (a)) revealed pathogen coverage is highly uneven:

| Pathogen | Continuous NHANES coverage |
|---|---|
| HCV | 2003–04 through 2021–23 — full series |
| HSV-1 / HSV-2 | 1999–2016 + 2017–March 2020 — full series |
| HPV serology | **2003–2010 only** (4 cycles); replaced by DNA-based testing from 2013–14 |
| EBV (VCA IgG) | **2003–2010 only**, children 6–19 |
| *H. pylori* IgG | **1999–2000 only** in continuous NHANES (plus NHANES III 1988–94) |

Only one window has all five pathogens co-measured in the same individuals.

## Decision

Use **NHANES 2003–2010 (four cycles, ~40,000 participants)** as the primary analysis window for the joint multi-pathogen serocatalytic model.

Secondary analyses:
- Long-run cohort-effect model for HCV + HSV-1 + HSV-2 only, spanning 1999–2023.
- *H. pylori*: single-cross-section catalytic fit on 1999–2000 (cohort effect implicit in age-prevalence shape, not estimated from time series).
- EBV: children-only arm, ages 6–19, 2003–2010.

## Consequences

**Positive:**
- Five-pathogen joint model is supported on a single coherent sample (the only window where this is true).
- Clean primary-analysis framing avoids fragile assumptions about cross-cycle assay comparability.
- Sample size (~40k) supports the multi-pathogen + interaction model planned in §7 of the project plan.

**Negative:**
- Cohort-effect estimation for HPV / EBV / *H. pylori* is limited; estimates rely on age-prevalence shape rather than time-series.
- HPV findings cannot speak to the post-2010 vaccine era within the serology framework (would require switching to DNA-based prevalence — a separate paper).
- Burden projections for *H. pylori* → gastric and EBV → nasopharyngeal must be flagged as resting on single-snapshot serology.

## Alternatives considered

1. **Use all available continuous NHANES cycles, accept missingness.** Rejected: the missingness pattern is structural (entire pathogens missing in entire cycles), not random; standard multiple imputation does not apply.
2. **Drop *H. pylori* and EBV entirely.** Rejected: both have strong cancer-attributable fractions; their inclusion is part of the paper's contribution. Keep as constrained sub-models with explicit caveats.
3. **Bridge to NHANES III (1988–94) for *H. pylori* cohort structure.** Deferred: requires assay-comparability work that adds weeks of validation; revisit in a follow-on paper if reviewers ask.

## References

- Project plan (this repository, latest version) — sections 7 and 11
- Artifact (a) — NHANES variable × cycle × pathogen access map (delivered 2026-05-30)
- Lewis et al., HPV trends in NHANES 2003–2010 — [PMC4687319](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4687319/)
