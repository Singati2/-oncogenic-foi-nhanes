"""Fit reverse-catalytic (seroreversion) model to NHANES HPV-16/18 (2003-2010).

Pipeline mirrors fit_hpv.py but estimates both lambda and omega.

Outputs:
    results/summary/hpv_reverse_summary.csv
    results/summary/hpv_constant_vs_reverse_per_age.csv
    results/summary/hpv_constant_vs_reverse_metrics.csv
    results/figures/hpv_constant_vs_reverse.png

Caveats (v0, MVP)
-----------------
- Unweighted exploratory seroreversion model.
- Sex-pooled.
- P(a) under reverse-catalytic is monotone non-decreasing; the observed
  decline at the oldest ages CANNOT be explained by seroreversion alone
  within this model class. See the audit report for the seroreversion vs
  cohort-effects identifiability caveat.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.data_pull.download import PROJECT_ROOT
from analysis.inference.fit_hpv import bin_by_age, filter_hpv
from analysis.model.bayesian_reverse import fit_reverse_catalytic
from analysis.model.catalytic import constant_foi_seroprevalence
from analysis.model.reverse_catalytic import reverse_catalytic_seroprevalence

logger = logging.getLogger(__name__)

PARQUET = PROJECT_ROOT / "data" / "interim" / "nhanes_2003_2010_oncogenic.parquet"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"


def fit_and_summarize(
    binned: pd.DataFrame,
    *,
    n_warmup: int,
    n_samples: int,
    seed: int,
) -> tuple[dict, dict]:
    """Run NUTS; return posterior samples dict and a summary dict."""
    mcmc = fit_reverse_catalytic(
        ages=binned["age_bin"].to_numpy(dtype=float),
        n_total=binned["n_total"].to_numpy(dtype=int),
        n_pos=binned["n_pos"].to_numpy(dtype=int),
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
    )
    raw = mcmc.get_samples()
    samples = {k: np.asarray(v) for k, v in raw.items()}

    def _summary(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "ci_lo_95": float(np.quantile(arr, 0.025)),
            "ci_hi_95": float(np.quantile(arr, 0.975)),
        }

    summary = {
        "lambda": _summary(samples["lambda"]),
        "omega": _summary(samples["omega"]),
        "steady_state": _summary(samples["steady_state"]),
    }
    return samples, summary


def per_age_comparison(
    binned: pd.DataFrame,
    samples: dict,
    constant_lambda: float,
) -> pd.DataFrame:
    """Per-age table: observed, constant pred, reverse pred (median+CrI), residuals."""
    out_rows = []
    for _, row in binned.iterrows():
        a = float(row["age_bin"])
        n = int(row["n_total"])
        n_pos = int(row["n_pos"])
        obs = n_pos / n
        pred_const = 1.0 - np.exp(-constant_lambda * a)
        # reverse pred posterior at age a
        sum_ = samples["lambda"] + samples["omega"]
        p_rev_draws = (samples["lambda"] / sum_) * (1.0 - np.exp(-sum_ * a))
        out_rows.append(
            {
                "age_bin": a,
                "n": n,
                "observed": obs,
                "obs_se": float(np.sqrt(obs * (1 - obs) / n)),
                "pred_constant": float(pred_const),
                "resid_constant": float(obs - pred_const),
                "pred_reverse_median": float(np.median(p_rev_draws)),
                "pred_reverse_lo": float(np.quantile(p_rev_draws, 0.025)),
                "pred_reverse_hi": float(np.quantile(p_rev_draws, 0.975)),
                "resid_reverse": float(obs - np.median(p_rev_draws)),
            }
        )
    df = pd.DataFrame(out_rows).sort_values("age_bin").reset_index(drop=True)
    df["resid_constant_per_se"] = df["resid_constant"] / df["obs_se"]
    df["resid_reverse_per_se"] = df["resid_reverse"] / df["obs_se"]
    return df


def plot_three_way(
    compared: pd.DataFrame,
    constant_lambda: float,
    samples: dict,
    out_path: Path,
) -> None:
    """Observed + constant-FOI curve + reverse-catalytic curve with 95% CrI."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)

    # Observed
    obs = compared["observed"]
    se = compared["obs_se"]
    ax.errorbar(
        compared["age_bin"], obs, yerr=1.96 * se,
        fmt="o", color="black", capsize=3, label="Observed (95% Wald)",
    )

    # Constant FOI
    age_grid = np.linspace(compared["age_bin"].min(), compared["age_bin"].max(), 200)
    ax.plot(
        age_grid, 1.0 - np.exp(-constant_lambda * age_grid),
        color="C3", lw=2, ls="--",
        label=f"Constant FOI (λ = {constant_lambda:.4f} yr⁻¹)",
    )

    # Reverse-catalytic posterior on the grid
    sum_draws = samples["lambda"] + samples["omega"]
    rev_curves = np.stack(
        [
            (samples["lambda"][s] / sum_draws[s])
            * (1.0 - np.exp(-sum_draws[s] * age_grid))
            for s in range(len(samples["lambda"]))
        ]
    )
    rev_lo, rev_med, rev_hi = np.quantile(rev_curves, [0.025, 0.5, 0.975], axis=0)
    lam_med = float(np.median(samples["lambda"]))
    om_med = float(np.median(samples["omega"]))
    ss_med = float(np.median(samples["steady_state"]))
    ax.plot(
        age_grid, rev_med,
        color="C0", lw=2,
        label=(
            f"Reverse-catalytic (λ̂={lam_med:.4f}, ω̂={om_med:.4f}, "
            f"steady state={ss_med:.3f})"
        ),
    )
    ax.fill_between(age_grid, rev_lo, rev_hi, alpha=0.20, color="C0",
                    label="Reverse-catalytic 95% CrI")

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("HPV-16/18 seroprevalence")
    ax.set_title("NHANES 2003-2010: constant-FOI vs reverse-catalytic")
    ax.set_ylim(0, max(0.30, rev_hi.max() * 1.1))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def load_constant_lambda_median() -> float | None:
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
        raise FileNotFoundError(f"{PARQUET} not found — run `make pull` first.")

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
    logger.info(
        "Posterior summary:\n  lambda:        %s\n  omega:         %s\n  steady_state:  %s",
        summary["lambda"], summary["omega"], summary["steady_state"],
    )

    constant_lambda = load_constant_lambda_median()
    if constant_lambda is None:
        raise RuntimeError(
            "constant-FOI summary missing at results/summary/hpv_lambda.csv; "
            "run `make fit-hpv` first."
        )

    compared = per_age_comparison(binned, samples, constant_lambda)
    rmse_const = float(np.sqrt((compared["resid_constant"] ** 2).mean()))
    rmse_rev = float(np.sqrt((compared["resid_reverse"] ** 2).mean()))
    logger.info("RMSE constant : %.4f", rmse_const)
    logger.info("RMSE reverse  : %.4f", rmse_rev)
    logger.info("Per-age comparison:\n%s",
                compared.to_string(index=False,
                                   formatters={c: "{:.4f}".format
                                               for c in compared.select_dtypes(float).columns}))

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"param": k, **v} for k, v in summary.items()
        ]
    ).to_csv(SUMMARY_DIR / "hpv_reverse_summary.csv", index=False)
    pd.DataFrame(
        [
            {"model": "constant", "rmse": rmse_const},
            {"model": "reverse", "rmse": rmse_rev},
        ]
    ).to_csv(SUMMARY_DIR / "hpv_constant_vs_reverse_metrics.csv", index=False)
    compared.to_csv(SUMMARY_DIR / "hpv_constant_vs_reverse_per_age.csv", index=False)

    fig_path = FIG_DIR / "hpv_constant_vs_reverse.png"
    plot_three_way(compared, constant_lambda, samples, fig_path)
    logger.info("wrote %s", fig_path)


if __name__ == "__main__":
    main()
