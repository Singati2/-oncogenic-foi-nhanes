"""Fit constant-FOI catalytic model to NHANES HPV-16/18 serology 2003-2010.

Loads the harmonized parquet, filters to participants with sero_hpv_hr measured,
bins by age, runs NUTS, and emits:

    results/summary/hpv_lambda.csv             posterior summary
    results/figures/hpv_posterior_predictive.png   observed vs fitted seroprev

Usage:
    python -m analysis.inference.fit_hpv
    python -m analysis.inference.fit_hpv --age-min 18 --age-max 59 --bin-width 5

Caveats (v0, MVP):
    - Sex-pooled (no stratification yet)
    - Unweighted likelihood (no survey weights yet)
    - Single FOI for the whole 2003-2010 window (no cohort/time structure)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.data_pull.download import PROJECT_ROOT
from analysis.model.bayesian import fit_catalytic
from analysis.model.catalytic import constant_foi_seroprevalence

logger = logging.getLogger(__name__)

PARQUET = PROJECT_ROOT / "data" / "interim" / "nhanes_2003_2010_oncogenic.parquet"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"


def filter_hpv(df: pd.DataFrame, age_min: float, age_max: float) -> pd.DataFrame:
    """Keep participants with non-missing sero_hpv_hr in the age window."""
    keep = (
        df["age"].between(age_min, age_max, inclusive="both")
        & df["sero_hpv_hr"].notna()
    )
    out = df.loc[keep, ["age", "sero_hpv_hr"]].copy()
    out["sero_hpv_hr"] = out["sero_hpv_hr"].astype(int)
    return out


def bin_by_age(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    """Aggregate (n_total, n_pos) per age bin; bin center used as the age."""
    binned = df.copy()
    binned["age_bin"] = (
        np.floor(binned["age"] / bin_width) * bin_width + bin_width / 2
    )
    out = (
        binned.groupby("age_bin", as_index=False)
        .agg(n_total=("sero_hpv_hr", "size"), n_pos=("sero_hpv_hr", "sum"))
        .sort_values("age_bin")
        .reset_index(drop=True)
    )
    return out


def fit_and_summarize(
    binned: pd.DataFrame,
    *,
    n_warmup: int,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Run NUTS; return posterior samples for lambda plus a summary dict."""
    mcmc = fit_catalytic(
        ages=binned["age_bin"].to_numpy(dtype=float),
        n_total=binned["n_total"].to_numpy(dtype=int),
        n_pos=binned["n_pos"].to_numpy(dtype=int),
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
    )
    samples = np.asarray(mcmc.get_samples()["lambda"])
    summary = {
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "ci_lo_95": float(np.quantile(samples, 0.025)),
        "ci_hi_95": float(np.quantile(samples, 0.975)),
        "n_samples": int(samples.size),
    }
    return samples, summary


def plot_posterior_predictive(
    binned: pd.DataFrame, samples: np.ndarray, out_path: Path
) -> None:
    """Observed seroprevalence per age bin vs posterior predictive curve."""
    obs_p = binned["n_pos"] / binned["n_total"]
    se = np.sqrt(obs_p * (1 - obs_p) / binned["n_total"])

    age_grid = np.linspace(
        binned["age_bin"].min(), binned["age_bin"].max(), 200
    )
    pred = np.stack(
        [constant_foi_seroprevalence(s, age_grid) for s in samples]
    )
    lo, med, hi = np.quantile(pred, [0.025, 0.5, 0.975], axis=0)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    ax.errorbar(
        binned["age_bin"],
        obs_p,
        yerr=1.96 * se,
        fmt="o",
        color="black",
        capsize=3,
        label="Observed (95% Wald)",
    )
    ax.plot(
        age_grid,
        med,
        color="C0",
        lw=2,
        label=f"Posterior median (λ̂ = {samples.mean():.4f} yr⁻¹)",
    )
    ax.fill_between(age_grid, lo, hi, alpha=0.20, color="C0", label="95% CrI")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("HPV-16/18 seroprevalence")
    ax.set_title("NHANES 2003-2010: constant-FOI catalytic fit (sex-pooled)")
    ax.set_ylim(0, max(0.5, hi.max() * 1.1))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--age-min", type=float, default=14.0)
    parser.add_argument("--age-max", type=float, default=59.0)
    parser.add_argument("--bin-width", type=float, default=5.0)
    parser.add_argument("--n-warmup", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if not PARQUET.exists():
        raise FileNotFoundError(
            f"{PARQUET} not found — run `make pull` first to build it."
        )

    df = pd.read_parquet(PARQUET)
    hpv = filter_hpv(df, args.age_min, args.age_max)
    logger.info("HPV-serology participants in [%.0f, %.0f]: %d",
                args.age_min, args.age_max, len(hpv))

    binned = bin_by_age(hpv, args.bin_width)
    logger.info("Age bins:\n%s", binned.to_string(index=False))

    samples, summary = fit_and_summarize(
        binned,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    logger.info("Posterior summary: %s", summary)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / "hpv_lambda.csv"
    pd.Series(summary).to_csv(summary_path, header=False)
    logger.info("wrote %s", summary_path)

    fig_path = FIG_DIR / "hpv_posterior_predictive.png"
    plot_posterior_predictive(binned, samples, fig_path)
    logger.info("wrote %s", fig_path)


if __name__ == "__main__":
    main()
