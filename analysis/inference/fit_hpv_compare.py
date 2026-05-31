"""WAIC/LOO comparison of constant-FOI vs reverse-catalytic, plus a drop-oldest-bin
sensitivity refit of the reverse model, on NHANES 2003-2010 HPV-16/18 serology.

What this script does and why it differs from fit_hpv.py / fit_hpv_reverse.py
----------------------------------------------------------------------------
1. Both models are refit jointly with `init_to_value` so that NUTS starts near
   the data-consistent posterior mode. Cold-start NUTS on the constant-FOI
   likelihood from HalfNormal(0.1) gets stuck for some chains in the broad
   prior tail (per-chain means split between ~0.005 and ~0.2, posterior CI
   spans the full prior support). The reverse-catalytic likelihood is gentler
   and converges cleanly from cold start, but we init it too for symmetry.

2. The Bayesian fit uses the BINNED likelihood (9 Binomial terms, one per
   age bin) — the posterior is the binned posterior. For WAIC/LOO we expand
   this into PARTICIPANT-LEVEL BERNOULLI terms: each of the ~14,478
   participants contributes one Bernoulli log-likelihood evaluated at their
   bin's predicted probability p_{b(i)}, NOT at their raw age. Within a bin
   every participant shares the same p. The expansion is mathematically
   equivalent to the Binomial up to a parameter-independent constant, so the
   posterior is unchanged; only the IC accounting changes.

   Why this matters for diagnostics: at bin level, leave-one-out removes
   ~1,500 participants per fold; PSIS importance sampling fails (Pareto k
   > 0.7 for most folds on the constant-FOI fit) and p_waic is inflated
   relative to the 1-2 actual model parameters. At participant granularity,
   LOO is a per-Bernoulli perturbation, Pareto k is well-controlled, and
   p_waic matches model complexity (~1 for constant, ~1.4 for reverse).

   Caveat: the WAIC/LOO produced here compares two BINNED models on a
   participant-level loss decomposition. It does NOT evaluate p at each
   participant's raw age. A genuinely age-resolved IC (refit with raw ages
   rather than bin centers, or evaluate p at each participant's raw age
   while keeping the binned fit) would be a separate robustness check.

3. The drop-oldest-bin reverse refit answers: when the age-57.5 bin (where the
   observed seroprevalence declines) is removed, does omega drop toward a
   biologically plausible value? This probes whether the model's preference for
   fast seroreversion is concentrated in the right-tail or distributed across
   age — relevant because the reverse-catalytic curve is monotone non-decreasing
   and so cannot bend down at older ages within model class.

Outputs
-------
    results/summary/hpv_waic_comparison.csv         WAIC ranking via az.compare
    results/summary/hpv_loo_comparison.csv          LOO  ranking via az.compare
    results/summary/hpv_reverse_drop_oldest_bin.csv full vs drop-57.5 posteriors

Usage
-----
    python -m analysis.inference.fit_hpv_compare
    python -m analysis.inference.fit_hpv_compare --n-chains 4 --n-samples 4000

Note on environment
-------------------
ArviZ's `_warn_once_per_day` does an atomic write into
`~/Library/Caches/arviz/` on macOS that crashes if the dir is missing on first
import. If you see a FileNotFoundError on `daily_warning.tmp`, run:

    mkdir -p ~/Library/Caches/arviz
    date -u +%Y-%m-%d > ~/Library/Caches/arviz/daily_warning
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.infer import MCMC, NUTS, init_to_value

from analysis.data_pull.download import PROJECT_ROOT
from analysis.inference.fit_hpv import bin_by_age, filter_hpv
from analysis.model.bayesian import catalytic_model
from analysis.model.bayesian_reverse import reverse_catalytic_model

logger = logging.getLogger(__name__)

PARQUET = PROJECT_ROOT / "data" / "interim" / "nhanes_2003_2010_oncogenic.parquet"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"


def run_mcmc(
    model_fn,
    ages,
    n_total,
    n_pos,
    init_values: dict,
    *,
    n_warmup: int,
    n_samples: int,
    n_chains: int,
    seed: int,
) -> MCMC:
    kernel = NUTS(model_fn, init_strategy=init_to_value(values=init_values))
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        progress_bar=False,
        chain_method="sequential",
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        ages=jnp.asarray(ages, dtype=jnp.float32),
        n_total=jnp.asarray(n_total, dtype=jnp.int32),
        n_pos=jnp.asarray(n_pos, dtype=jnp.int32),
    )
    return mcmc


def _expand_to_participants(n_total: np.ndarray, n_pos: np.ndarray):
    """Construct participant-level (bin_index, y) for sum(n_total) participants.

    Within each bin, the first n_pos[i] participants are seropositive (y=1),
    the remaining n_total[i] - n_pos[i] are seronegative (y=0). Within-bin
    participant order does not affect WAIC/LOO since they all share p_{b(i)}.
    """
    n_total = np.asarray(n_total, dtype=int)
    n_pos = np.asarray(n_pos, dtype=int)
    bin_idx = np.repeat(np.arange(len(n_total)), n_total)
    y = np.zeros(int(n_total.sum()), dtype=int)
    cursor = 0
    for i, (nt, k) in enumerate(zip(n_total, n_pos)):
        y[cursor:cursor + int(k)] = 1
        cursor += int(nt)
    return bin_idx, y


def _ll_per_participant(p_bin: np.ndarray, bin_idx: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log Bernoulli(y_i ; p_{b(i)}) per draw, per participant.

    Each participant inherits their bin's predicted probability — within-bin
    all participants share p_{b(i)}, not a raw-age-specific value.

    p_bin:    (n_draws, n_bins)
    bin_idx:  (n_participants,)
    y:        (n_participants,)
    returns:  (n_draws, n_participants)
    """
    p_part = p_bin[:, bin_idx]
    eps = 1e-300
    log_p = np.log(np.clip(p_part, eps, 1.0))
    log_1mp = np.log(np.clip(1.0 - p_part, eps, 1.0))
    return np.where(y[None, :] == 1, log_p, log_1mp)


def participant_bernoulli_loglik_constant(
    samples: dict, ages: np.ndarray, n_total: np.ndarray, n_pos: np.ndarray
) -> np.ndarray:
    """Bin-pooled p expanded to one Bernoulli log-lik per participant (constant-FOI).

    For each bin i: p_i = 1 - exp(-lambda * age_bin_i). Every participant in
    bin i contributes log Bernoulli(y; p_i) — they share the bin's predicted
    probability, not a raw-age-specific value.
    """
    bin_idx, y = _expand_to_participants(n_total, n_pos)
    lam = np.asarray(samples["lambda"])
    p_bin = 1.0 - np.exp(-lam[:, None] * ages[None, :])
    return _ll_per_participant(p_bin, bin_idx, y)


def participant_bernoulli_loglik_reverse(
    samples: dict, ages: np.ndarray, n_total: np.ndarray, n_pos: np.ndarray
) -> np.ndarray:
    """Bin-pooled p expanded to one Bernoulli log-lik per participant (reverse-catalytic).

    For each bin i:
        p_i = (lambda / (lambda + omega)) * (1 - exp(-(lambda + omega) * age_bin_i)).
    Every participant in bin i contributes log Bernoulli(y; p_i) — they share
    the bin's predicted probability, not a raw-age-specific value.
    """
    bin_idx, y = _expand_to_participants(n_total, n_pos)
    lam = np.asarray(samples["lambda"])
    om = np.asarray(samples["omega"])
    s = lam + om
    p_bin = (lam[:, None] / s[:, None]) * (1.0 - np.exp(-s[:, None] * ages[None, :]))
    return _ll_per_participant(p_bin, bin_idx, y)


def build_idata(
    mcmc: MCMC,
    ll_participant: np.ndarray,
    *,
    n_chains: int,
    n_samples: int,
) -> az.InferenceData:
    posterior = {
        k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()
    }
    n_participants = ll_participant.shape[1]
    ll = ll_participant.reshape(n_chains, n_samples, n_participants)
    return az.from_dict(posterior=posterior, log_likelihood={"obs": ll})


def summarize(mcmc: MCMC, params: list[str]) -> dict:
    raw = mcmc.get_samples()
    out = {}
    for p in params:
        arr = np.asarray(raw[p])
        out[p] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "ci_lo_95": float(np.quantile(arr, 0.025)),
            "ci_hi_95": float(np.quantile(arr, 0.975)),
        }
    return out


def chain_diagnostics(mcmc: MCMC, name: str) -> None:
    samples = mcmc.get_samples(group_by_chain=True)
    print(f"  --- {name} per-chain diagnostics ---")
    for p, arr in samples.items():
        arr_np = np.asarray(arr)  # (chains, draws)
        means = ", ".join(f"{arr_np[c].mean():.5f}" for c in range(arr_np.shape[0]))
        print(
            f"    {p}: per-chain means = [{means}], "
            f"ESS bulk ≈ {int(az.ess(arr_np))}, "
            f"R-hat ≈ {float(az.rhat(arr_np)):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--age-min", type=float, default=14.0)
    parser.add_argument("--age-max", type=float, default=59.0)
    parser.add_argument("--bin-width", type=float, default=5.0)
    parser.add_argument("--n-warmup", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument(
        "--drop-oldest-threshold",
        type=float,
        default=57.0,
        help="Drop bins with age_bin >= this threshold for the sensitivity refit.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if not PARQUET.exists():
        raise FileNotFoundError(
            f"{PARQUET} not found — run `make pull` first to build it."
        )

    df = pd.read_parquet(PARQUET)
    hpv = filter_hpv(df, args.age_min, args.age_max)
    binned = bin_by_age(hpv, args.bin_width)
    n_participants = int(binned["n_total"].sum())
    print(f"\n=== Binned data ({len(binned)} bins, n={n_participants}) ===")
    print(binned.to_string(index=False))

    ages = binned["age_bin"].to_numpy(dtype=float)
    n_total = binned["n_total"].to_numpy(dtype=int)
    n_pos = binned["n_pos"].to_numpy(dtype=int)

    # ---- Constant-FOI ----
    print("\n=== Constant-FOI (init_to_value lambda=0.005) ===")
    mcmc_const = run_mcmc(
        catalytic_model, ages, n_total, n_pos,
        init_values={"lambda": 0.005},
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, seed=args.seed,
    )
    summ_const = summarize(mcmc_const, ["lambda"])
    print(json.dumps(summ_const, indent=2))
    chain_diagnostics(mcmc_const, "constant-FOI")
    ll_const = participant_bernoulli_loglik_constant(
        mcmc_const.get_samples(), ages, n_total, n_pos
    )
    idata_const = build_idata(
        mcmc_const, ll_const,
        n_chains=args.n_chains, n_samples=args.n_samples,
    )

    # ---- Reverse-catalytic (full data) ----
    print("\n=== Reverse-catalytic (init_to_value lambda=0.01, omega=0.05) ===")
    mcmc_rev = run_mcmc(
        reverse_catalytic_model, ages, n_total, n_pos,
        init_values={"lambda": 0.01, "omega": 0.05},
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, seed=args.seed,
    )
    summ_rev = summarize(mcmc_rev, ["lambda", "omega", "steady_state"])
    print(json.dumps(summ_rev, indent=2))
    chain_diagnostics(mcmc_rev, "reverse")
    ll_rev = participant_bernoulli_loglik_reverse(
        mcmc_rev.get_samples(), ages, n_total, n_pos
    )
    idata_rev = build_idata(
        mcmc_rev, ll_rev,
        n_chains=args.n_chains, n_samples=args.n_samples,
    )

    # ---- WAIC/LOO (binned likelihood, participant-Bernoulli expansion) ----
    print(
        f"\n=== WAIC (binned likelihood, participant-Bernoulli expansion; "
        f"n={n_participants}) ==="
    )
    print("constant:", az.waic(idata_const, scale="log"))
    print("reverse: ", az.waic(idata_rev, scale="log"))

    print(
        f"\n=== LOO  (binned likelihood, participant-Bernoulli expansion; "
        f"n={n_participants}) ==="
    )
    print("constant:", az.loo(idata_const, scale="log"))
    print("reverse: ", az.loo(idata_rev, scale="log"))

    print("\n=== az.compare WAIC ===")
    cmp_w = az.compare(
        {"constant": idata_const, "reverse": idata_rev}, ic="waic", scale="log"
    )
    print(cmp_w.to_string())

    print("\n=== az.compare LOO ===")
    cmp_l = az.compare(
        {"constant": idata_const, "reverse": idata_rev}, ic="loo", scale="log"
    )
    print(cmp_l.to_string())

    # ---- Drop-oldest sensitivity ----
    mask = ages < args.drop_oldest_threshold
    ages_d = ages[mask]
    n_total_d = n_total[mask]
    n_pos_d = n_pos[mask]
    print(
        f"\n=== Drop-oldest reverse refit: kept {len(ages_d)} bins "
        f"(ages {ages_d.tolist()}) ==="
    )
    mcmc_rev_drop = run_mcmc(
        reverse_catalytic_model, ages_d, n_total_d, n_pos_d,
        init_values={"lambda": 0.01, "omega": 0.05},
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, seed=args.seed,
    )
    summ_rev_drop = summarize(
        mcmc_rev_drop, ["lambda", "omega", "steady_state"]
    )
    print(json.dumps(summ_rev_drop, indent=2))
    chain_diagnostics(mcmc_rev_drop, "reverse (drop oldest)")

    # ---- Write outputs ----
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    cmp_w.to_csv(SUMMARY_DIR / "hpv_waic_comparison.csv")
    cmp_l.to_csv(SUMMARY_DIR / "hpv_loo_comparison.csv")
    drop_df = pd.DataFrame(
        [
            {"param": p, "data": "full", **summ_rev[p]}
            for p in ["lambda", "omega", "steady_state"]
        ]
        + [
            {"param": p, "data": "drop_oldest", **summ_rev_drop[p]}
            for p in ["lambda", "omega", "steady_state"]
        ]
    )
    drop_df.to_csv(SUMMARY_DIR / "hpv_reverse_drop_oldest_bin.csv", index=False)

    print(f"\nWrote {SUMMARY_DIR / 'hpv_waic_comparison.csv'}")
    print(f"Wrote {SUMMARY_DIR / 'hpv_loo_comparison.csv'}")
    print(f"Wrote {SUMMARY_DIR / 'hpv_reverse_drop_oldest_bin.csv'}")


if __name__ == "__main__":
    main()
