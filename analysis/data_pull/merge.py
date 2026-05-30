"""Stack harmonized per-cycle tables into a single analysis-ready dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from analysis.data_pull.download import PROJECT_ROOT, download_all, load_manifest
from analysis.data_pull.harmonize import harmonize_cycle

logger = logging.getLogger(__name__)

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUT_NAME = "nhanes_2003_2010_oncogenic.parquet"


def build_analysis_table(*, cycles: list[str] | None = None) -> pd.DataFrame:
    """End-to-end: download → harmonize → stack → write parquet."""
    spec = load_manifest()
    paths_by_cycle = download_all(cycles=cycles)

    frames = []
    for code, paths in paths_by_cycle.items():
        logger.info("harmonizing cycle %s", code)
        frames.append(harmonize_cycle(code, paths, spec))

    pooled = pd.concat(frames, ignore_index=True)

    # NHANES analytic guideline for multi-cycle pooling: divide WTMEC2YR by N cycles.
    n_cycles = len(paths_by_cycle)
    pooled["weight_mec_pooled"] = pooled["weight_mec_2y"] / n_cycles

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / OUT_NAME
    pooled.to_parquet(out_path, index=False)
    logger.info("wrote %s (rows=%d, cols=%d)", out_path, len(pooled), pooled.shape[1])

    _print_coverage_summary(pooled)
    return pooled


def _print_coverage_summary(df: pd.DataFrame) -> None:
    """Log per-cycle non-missing counts for each sero variable — sanity check."""
    sero_cols = [c for c in df.columns if c.startswith("sero_")]
    summary = df.groupby("cycle")[sero_cols].agg(lambda s: s.notna().sum())
    logger.info("non-missing serostatus counts by cycle:\n%s", summary.to_string())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    build_analysis_table()


if __name__ == "__main__":
    main()
