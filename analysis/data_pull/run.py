"""CLI entry point: download → harmonize → stack → write parquet.

    python -m analysis.data_pull.run             # all cycles (C, D, E, F)
    python -m analysis.data_pull.run --cycles C  # one cycle (faster smoke test)
    python -m analysis.data_pull.run -v          # verbose logging
"""

from __future__ import annotations

import argparse
import logging

from analysis.data_pull.merge import build_analysis_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NHANES 2003-2010 oncogenic-virus serology pipeline"
    )
    parser.add_argument(
        "--cycles",
        nargs="+",
        choices=["C", "D", "E", "F"],
        help="Subset of cycles (default: all four)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    build_analysis_table(cycles=args.cycles)


if __name__ == "__main__":
    main()
