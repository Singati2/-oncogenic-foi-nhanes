"""Network- and NUTS-free unit tests for the inference helpers.

Tests the pure-function pieces of `analysis/inference/fit_hpv.py` —
`filter_hpv` and `bin_by_age` — using synthetic dataframes. The actual
NUTS fit is covered by `test_parameter_recovery.py` and the model smoke
test, so we don't repeat it here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.inference.fit_hpv import bin_by_age, filter_hpv


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 12, 15, 20, 25, 30, 50, 60, 65],
            "sero_hpv_hr": pd.array(
                [1, 0, 1, 0, 1, pd.NA, 0, pd.NA, 1], dtype="Int8"
            ),
        }
    )


def test_filter_hpv_drops_missing_and_out_of_age() -> None:
    df = _make_df()
    out = filter_hpv(df, age_min=14, age_max=59)
    # ages kept: 15, 20, 25, 30, 50  (30 has NA -> dropped; <14 and >59 dropped)
    assert sorted(out["age"].tolist()) == [15, 20, 25, 50]


def test_filter_hpv_returns_int_serostatus() -> None:
    df = _make_df()
    out = filter_hpv(df, age_min=10, age_max=99)
    assert out["sero_hpv_hr"].dtype.kind == "i"
    assert set(out["sero_hpv_hr"].unique()) <= {0, 1}


def test_bin_by_age_aggregates_counts() -> None:
    df = pd.DataFrame(
        {
            "age": [15, 17, 19, 22, 25, 30, 31],
            "sero_hpv_hr": [1, 0, 1, 1, 0, 1, 0],
        }
    )
    out = bin_by_age(df, bin_width=10.0)
    # floor(age/10)*10 + 5 :
    #   15/17/19 -> bin 15  (3 obs, pos: 1,0,1 -> 2)
    #   22/25    -> bin 25  (2 obs, pos: 1,0   -> 1)
    #   30/31    -> bin 35  (2 obs, pos: 1,0   -> 1)
    expected = pd.DataFrame(
        {
            "age_bin": [15.0, 25.0, 35.0],
            "n_total": [3, 2, 2],
            "n_pos": [2, 1, 1],
        }
    )
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_bin_by_age_sorted_by_age() -> None:
    df = pd.DataFrame({"age": [50, 20, 30], "sero_hpv_hr": [0, 1, 1]})
    out = bin_by_age(df, bin_width=10.0)
    assert (np.diff(out["age_bin"].to_numpy()) > 0).all()


# ---------- bin_by_age hardening: determinism / NaN / empty-bin guarantees ----------

def test_bin_by_age_is_bit_for_bit_deterministic() -> None:
    """50 repeated invocations on the same input yield identical output bytes."""
    rng = np.random.default_rng(20260530)
    df = pd.DataFrame(
        {
            "age": rng.integers(10, 80, 500).astype(float),
            "sero_hpv_hr": rng.integers(0, 2, 500),
        }
    )
    first = bin_by_age(df.copy(), bin_width=5.0)
    for _ in range(50):
        again = bin_by_age(df.copy(), bin_width=5.0)
        pd.testing.assert_frame_equal(first, again, check_exact=True)


def test_bin_by_age_drops_nan_age() -> None:
    df = pd.DataFrame(
        {
            "age": [20.0, np.nan, 30.0],
            "sero_hpv_hr": [1, 1, 0],
        }
    )
    out = bin_by_age(df, bin_width=5.0)
    assert out["n_total"].sum() == 2
    assert sorted(out["age_bin"].tolist()) == [22.5, 32.5]


def test_bin_by_age_drops_nan_sero() -> None:
    df = pd.DataFrame(
        {
            "age": [20.0, 25.0, 30.0, 35.0],
            "sero_hpv_hr": pd.array([1, pd.NA, 1, 0], dtype="Int8"),
        }
    )
    out = bin_by_age(df, bin_width=5.0)
    # the NA-sero row at age 25 is dropped; three rows survive
    assert out["n_total"].sum() == 3


def test_bin_by_age_returns_no_empty_bins() -> None:
    """No row in the output has n_total == 0."""
    df = pd.DataFrame(
        {"age": [20, 21, 50, 51], "sero_hpv_hr": [1, 0, 1, 0]}
    )
    out = bin_by_age(df, bin_width=5.0)
    assert (out["n_total"] > 0).all()
    assert len(out) == 2  # bins 22.5 and 52.5 only


def test_bin_by_age_exact_bin_count_for_known_input() -> None:
    """Bin count equals the number of distinct 5-yr strips covered by the data."""
    df = pd.DataFrame(
        {"age": [15, 17, 19, 22, 25, 30, 31], "sero_hpv_hr": [1, 0, 1, 1, 0, 1, 0]}
    )
    out = bin_by_age(df, bin_width=10.0)
    assert len(out) == 3
    assert out["age_bin"].tolist() == [15.0, 25.0, 35.0]
