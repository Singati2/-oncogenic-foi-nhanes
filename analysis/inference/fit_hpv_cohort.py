"""Fit cohort-structured catalytic model to NHANES HPV-16/18 (2003-2010).

Pipeline
--------
1. Load `data/interim/nhanes_2003_2010_oncogenic.parquet`
2. Filter to non-missing `sero_hpv_hr` in [age_min, age_max]
3. Derive birth_year = CYCLE_MIDPOINT[cycle] - age
4. Assign birth cohort: pre-1955 / 1955-1969 / 1970-1984 / post-1984
5. Report per-cohort n and crude prevalence
6. Aggregate per (age_bin, cohort) and fit hierarchical NumPyro model
7. Emit:
       results/summary/hpv_lambda_by_cohort.csv
       results/summary/hpv_cohort_fit_metrics.csv
       results/figures/hpv_cohort_vs_constant.png

Caveats (v0, MVP)
-----------------
- Unweighted exploratory cohort model (no survey weights yet)
- Sex-pooled
- Constant FOI WITHIN each cohort (no within-cohort age structure)
- No seroreversion / antibody waning
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.data_pull.download import PROJECT_ROOT
from analysis.model.bayesian_cohort import fit_cohort_catalytic
from analysis.model.catalytic import constant_foi_seroprevalence

logger = logging.getLogger(__name__)

PARQUET = PROJECT_ROOT / "data" / "interim" / "nhanes_2003_2010_oncogenic.parquet"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"

CYCLE_MIDPOINT: dict[str, int] = {"C": 2004, "D": 2006, "E": 2008, "F": 2010}

COHORT_LABELS: list[str] = ["pre-1955", "1955-1969", "1970-1984", "post-1984"]
COHORT_BOUNDS: list[float] = [-np.inf, 1955.0, 1970.0, 1985.0, np.inf]


def prepare(df: pd.DataFrame, age_min: float, age_max: float) -> pd.DataFrame:
    """Filter to non-missing HPV serology, derive birth year and cohort."""
    keep = (
        df["age"].between(age_min, age_max, inclusive="both")
        & df["sero_hpv_hr"].notna()
    )
    out = df.loc[keep, ["cycle", "age", "sero_hpv_hr"]].copy()
    out["sero_hpv_hr"] = out["sero_hpv_hr"].astype(int)
    out["birth_year"] = out["cycle"].map(CYCLE_MIDPOINT) - out["age"]
    out["cohort"] = pd.cut(
        out["birth_year"],
        bins=COHORT_BOUNDS,
        labels=COHORT_LABELS,
        right=False,
        ordered=True,
    )
    return out


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cohort n, n_pos, crude prevalence, observed age and birth-year range."""
    s = (
        df.groupby("cohort", observed=True)
        .agg(
            n=("sero_hpv_hr", "size"),
            n_pos=("sero_hpv_hr", "sum"),
            prev=("sero_hpv_hr", "mean"),
            age_min=("age", "min"),
            age_max=("age", "max"),
            birth_min=("birth_year", "min"),
            birth_max=("birth_year", "max"),
        )
        .reset_index()
    )
    return s


def bin_age_cohort(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    """Aggregate per (age_bin, cohort); drop empty cells."""
    work = df.copy()
    work["age_bin"] = (
        np.floor(work["age"] / bin_width) * bin_width + bin_width / 2
    )
    out = (
        work.groupby(["age_bin", "cohort"], observed=True, as_index=False)
        .agg(n_total=("sero_hpv_hr", "size"), n_pos=("sero_hpv_hr", "sum"))
    )
    out = out[out["n_total"] > 0].sort_values(["age_bin", "cohort"]).reset_index(drop=True)
    return out


def fit_and_summarize(
    binned: pd.DataFrame,
    cohort_labels: list[str],
    *,
    n_warmup: int,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run NUTS; return posterior lambda samples (n_samples, n_cohorts) and a summary."""
    cohort_to_idx = {c: i for i, c in enumerate(cohort_labels)}
    cohort_idx = binned["cohort"].map(cohort_to_idx).to_numpy(dtype=int)

    mcmc = fit_cohort_catalytic(
        ages=binned["age_bin"].to_numpy(dtype=float),
        cohort_idx=cohort_idx,
        n_cohorts=len(cohort_labels),
        n_total=binned["n_total"].to_numpy(dtype=int),
        n_pos=binned["n_pos"].to_numpy(dtype=int),
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
    )
    samples = np.asarray(mcmc.get_samples()["lambda"])  # shape (S, K)

    rows = []
    for k, label in enumerate(cohort_labels):
        s = samples[:, k]
        rows.append(
            {
                "cohort": label,
                "mean": float(s.mean()),
                "median": float(np.median(s)),
                "ci_lo_95": float(np.quantile(s, 0.025)),
                "ci_hi_95": float(np.quantile(s, 0.975)),
            }
        )
    return samples, pd.DataFrame(rows)


def per_age_pooled_prediction(
    binned: pd.DataFrame,
    samples: np.ndarray,
    cohort_labels: list[str],
) -> pd.DataFrame:
    """Cohort-composition-weighted pooled prediction per age bin.

    For each age_bin a:  predicted(a) = sum_k w_{a,k} * (1 - exp(-lambda_k * a))
    where w_{a,k} = n_{a,k} / sum_k n_{a,k}.

    Returns a DataFrame with age_bin, n, observed, pred_median, pred_lo, pred_hi.
    """
    cohort_to_idx = {c: i for i, c in enumerate(cohort_labels)}

    rows = []
    for age_bin, sub in binned.groupby("age_bin"):
        n_total = int(sub["n_total"].sum())
        n_pos = int(sub["n_pos"].sum())
        weights = sub["n_total"].to_numpy() / n_total
        idxs = np.array([cohort_to_idx[c] for c in sub["cohort"]])

        # samples_for_bin: shape (S, len(sub))
        lam = samples[:, idxs]  # broadcasting over posterior draws
        p_per = 1.0 - np.exp(-lam * float(age_bin))   # shape (S, len(sub))
        p_pooled = (p_per * weights[None, :]).sum(axis=1)  # shape (S,)

        rows.append(
            {
                "age_bin": float(age_bin),
                "n": n_total,
                "observed": n_pos / n_total,
                "pred_median": float(np.median(p_pooled)),
                "pred_lo": float(np.quantile(p_pooled, 0.025)),
                "pred_hi": float(np.quantile(p_pooled, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values("age_bin").reset_index(drop=True)


def compare_with_constant(
    per_age: pd.DataFrame, constant_lambda: float
) -> pd.DataFrame:
    """Add constant-model prediction and residuals; compute RMSE for both."""
    out = per_age.copy()
    out["pred_constant"] = 1.0 - np.exp(-constant_lambda * out["age_bin"])
    out["resid_cohort"] = out["observed"] - out["pred_median"]
    out["resid_constant"] = out["observed"] - out["pred_constant"]
    return out


def plot_comparison(
    per_age: pd.DataFrame,
    constant_lambda: float,
    samples: np.ndarray,
    cohort_labels: list[str],
    binned: pd.DataFrame,
    out_path: Path,
) -> None:
    """Pooled-by-age comparison: observed, constant fit, cohort-pooled fit."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)

    # Observed
    obs = per_age["observed"]
    se = np.sqrt(obs * (1 - obs) / per_age["n"])
    ax.errorbar(
        per_age["age_bin"], obs, yerr=1.96 * se,
        fmt="o", color="black", capsize=3, label="Observed (95% Wald)",
    )

    # Constant FOI
    age_grid = np.linspace(per_age["age_bin"].min(), per_age["age_bin"].max(), 200)
    pred_const = 1.0 - np.exp(-constant_lambda * age_grid)
    ax.plot(
        age_grid, pred_const,
        color="C3", lw=2, ls="--",
        label=f"Constant FOI (λ = {constant_lambda:.4f} yr⁻¹)",
    )

    # Cohort-pooled posterior predictive at the observed age bins
    ax.plot(
        per_age["age_bin"], per_age["pred_median"],
        color="C0", lw=2, marker="s",
        label="Cohort model (posterior median, pooled)",
    )
    ax.fill_between(
        per_age["age_bin"], per_age["pred_lo"], per_age["pred_hi"],
        alpha=0.20, color="C0", label="Cohort 95% CrI",
    )

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("HPV-16/18 seroprevalence")
    ax.set_title("NHANES 2003-2010: constant vs cohort-structured catalytic")
    ax.set_ylim(0, max(0.30, per_age["pred_hi"].max() * 1.1))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def load_constant_lambda() -> float | None:
    """Read the previously fitted constant-FOI lambda for comparison."""
    p = SUMMARY_DIR / "hpv_lambda.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p, header=None, index_col=0).squeeze("columns")
    return float(s["median"])


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
            f"{PARQUET} not found — run `make pull` first."
        )

    df = pd.read_parquet(PARQUET)
    hpv = prepare(df, args.age_min, args.age_max)
    logger.info("N HPV-serology rows in [%.0f, %.0f]: %d",
                args.age_min, args.age_max, len(hpv))

    cs = cohort_summary(hpv)
    logger.info("Per-cohort summary:\n%s",
                cs.to_string(index=False,
                             formatters={"prev": "{:.4f}".format}))

    binned = bin_age_cohort(hpv, args.bin_width)
    logger.info("(age, cohort) cells fit: %d", len(binned))

    samples, post_summary = fit_and_summarize(
        binned,
        cohort_labels=COHORT_LABELS,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    logger.info("Posterior lambda per cohort:\n%s",
                post_summary.to_string(index=False))

    per_age = per_age_pooled_prediction(binned, samples, COHORT_LABELS)

    constant_lambda = load_constant_lambda()
    if constant_lambda is None:
        logger.warning(
            "constant-FOI summary missing; "
            "comparison RMSE will be cohort-only."
        )
        constant_lambda = float("nan")

    compared = compare_with_constant(per_age, constant_lambda)
    rmse_constant = float(np.sqrt((compared["resid_constant"] ** 2).mean()))
    rmse_cohort = float(np.sqrt((compared["resid_cohort"] ** 2).mean()))
    logger.info("RMSE constant : %.4f", rmse_constant)
    logger.info("RMSE cohort   : %.4f", rmse_cohort)
    logger.info("Per-age comparison:\n%s",
                compared.to_string(index=False,
                                   formatters={c: "{:.4f}".format
                                               for c in ["observed", "pred_median",
                                                         "pred_lo", "pred_hi",
                                                         "pred_constant",
                                                         "resid_cohort", "resid_constant"]}))

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    post_summary.to_csv(SUMMARY_DIR / "hpv_lambda_by_cohort.csv", index=False)
    pd.DataFrame(
        [
            {"model": "constant", "rmse": rmse_constant},
            {"model": "cohort", "rmse": rmse_cohort},
        ]
    ).to_csv(SUMMARY_DIR / "hpv_cohort_fit_metrics.csv", index=False)
    compared.to_csv(SUMMARY_DIR / "hpv_cohort_per_age_table.csv", index=False)

    fig_path = FIG_DIR / "hpv_cohort_vs_constant.png"
    plot_comparison(
        per_age=compared,
        constant_lambda=constant_lambda,
        samples=samples,
        cohort_labels=COHORT_LABELS,
        binned=binned,
        out_path=fig_path,
    )
    logger.info("wrote %s", fig_path)


if __name__ == "__main__":
    main()
