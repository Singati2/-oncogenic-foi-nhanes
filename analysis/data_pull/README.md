# `analysis/data_pull/` — NHANES download & harmonization pipeline

End-to-end pipeline producing the analysis-ready table for the multi-pathogen
serocatalytic model.

## What it does

1. **Download** — fetches NHANES 2003-2010 XPT files (demographics + HCV + HSV + HPV serology + EBV) from the CDC. Idempotent; cached under `data/raw/`.
2. **Harmonize** — reads each cycle's XPT files, applies the variable mappings in [`manifest.yaml`](manifest.yaml), and derives one row per participant with standardized seropositive flags (`sero_hcv`, `sero_hsv1`, `sero_hsv2`, `sero_hpv_hr`, `sero_ebv`).
3. **Merge** — stacks the four cycles and writes `data/interim/nhanes_2003_2010_oncogenic.parquet`.

## Run

```bash
make pull                                    # full pipeline
python -m analysis.data_pull.run --cycles C  # one cycle smoke test
python -m analysis.data_pull.run -v          # verbose logging
```

## Output schema

| column | type | meaning |
|---|---|---|
| `seqn` | int64 | NHANES respondent ID (join key) |
| `cycle` | str | "C" / "D" / "E" / "F" |
| `age` | float | RIDAGEYR — age in years |
| `sex` | int | 1 = male, 2 = female |
| `race` | int | RIDRETH1 |
| `education` | int | DMDEDUC2 (adults) |
| `poverty_ratio` | float | INDFMPIR |
| `weight_mec_2y` | float | WTMEC2YR — single-cycle weight |
| `weight_mec_pooled` | float | WTMEC2YR / N cycles — for the pooled analysis |
| `psu`, `stratum` | int | survey design |
| `sero_hcv`, `sero_hsv1`, `sero_hsv2`, `sero_hpv_hr`, `sero_ebv` | Int8 | 1 = positive, 0 = negative, NA = missing/indeterminate |

## Updating the manifest

CDC sometimes renames files or revises variable codings between cycles. If a
download fails or a sero column is unexpectedly all-NA:

1. Open the relevant cycle's laboratory catalog:  
   `https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=YYYY-YYYY`
2. Confirm the file name and variable codebook.
3. Edit [`manifest.yaml`](manifest.yaml) — no Python code change needed.
4. Re-run `make pull`.

## Design notes

- **No URLs in Python code.** All paths and variable names live in
  [`manifest.yaml`](manifest.yaml). Maintenance is a YAML edit, not a code
  change.
- **Idempotent downloads.** Existing non-empty files are reused; safe to re-run.
- **Defensive harmonization.** Missing files or unknown variable names log a
  warning and produce NA, rather than crashing the whole pipeline.
- **NHANES analytic compliance.** Multi-cycle pooled weights follow the NHANES
  guideline (`WTMEC2YR / N`).
