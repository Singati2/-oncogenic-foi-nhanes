# ADR 0002 — HPV serology limited to types 6, 11, 16, 18

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** project lead

## Context

The original project plan assumed the NHANES HPV serology panel covered the standard high-risk oncogenic types (16, 18, 31, 33, 35, 45, 52, 58). Inspection of the actual `L52SER_C`, `HPVSER_D`, `HPVSER_E`, and `HPVSER_F` files shows NHANES 2003–2010 measured **only four HPV types**:

| Type | Risk class | NHANES variable (C/D) | NHANES variable (E/F) |
|---|---|---|---|
| HPV-6  | low-risk (wart-associated) | `LBXS06MK` | `LBX06` |
| HPV-11 | low-risk (wart-associated) | `LBXS11MK` | `LBX11` |
| HPV-16 | **high-risk (oncogenic)** | `LBXS16MK` | `LBX16` |
| HPV-18 | **high-risk (oncogenic)** | `LBXS18MK` | `LBX18` |

Variable naming convention shifted between cycles D and E.

## Decision

The "high-risk HPV seropositive" derivation in this project uses **HPV-16 and HPV-18 only**.

The manifest encodes the per-cycle variable list in `variables.hpv.high_risk_type_vars_by_cycle`:

```yaml
high_risk_type_vars_by_cycle:
  C: [LBXS16MK, LBXS18MK]
  D: [LBXS16MK, LBXS18MK]
  E: [LBX16, LBX18]
  F: [LBX16, LBX18]
```

## Consequences

**Positive:**
- HPV-16 + HPV-18 account for **~70% of cervical cancer cases globally** (and a higher share of other HPV-attributable cancers — anal, oropharyngeal). The model still captures the dominant oncogenic signal.
- Variable-name shifts across cycles are absorbed by the manifest with no code change.

**Negative:**
- We cannot speak to types 31/33/45/52/58, which collectively add another ~20% of cervical-cancer burden. Sensitivity analyses cannot vary these.
- Burden projections must be framed as **HPV-16/18-attributable** rather than total HPV.

## Alternatives considered

1. **Use DNA-based HPV files from 2013–18 to access more types.** Rejected for the primary paper because (a) DNA prevalence ≠ serology cumulative-exposure, requiring a different model class; (b) most DNA files require RDC access; (c) DNA cycles don't overlap with the rest of the 2003–2010 sweet spot.
2. **Drop HPV from the joint model.** Rejected — HPV-16/18 is the most policy-relevant oncogenic infection (cervical-cancer elimination targets) and is one of the strongest motivators of the paper.

## References

- Variable inspection: `pyreadstat.read_xport()` of all four `*HPVSER_*` files, 2026-05-30
- IARC/de Martel attributable-fraction estimates for HPV-16/18 vs other HR types
