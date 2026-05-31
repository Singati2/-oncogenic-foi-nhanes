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
