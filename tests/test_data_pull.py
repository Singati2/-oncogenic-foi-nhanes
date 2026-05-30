"""Network-free tests for the data-pull pipeline.

Focus on:
- Manifest structural integrity (cycles, file specs, variable specs all present)
- Harmonization logic on synthetic inputs (no XPT files, no internet)
"""

from __future__ import annotations

import pandas as pd

from analysis.data_pull.download import load_manifest
from analysis.data_pull.harmonize import _coerce_sero, _derive_hpv_hr


# ---------- Manifest integrity ----------

def test_manifest_loads_with_four_cycles() -> None:
    m = load_manifest()
    assert "cycles" in m
    assert set(m["cycles"].keys()) == {"C", "D", "E", "F"}


def test_every_cycle_has_all_five_pathogen_files() -> None:
    m = load_manifest()
    required = {"demo", "hcv", "hsv", "hpv", "ebv"}
    for code, cycle in m["cycles"].items():
        assert set(cycle["files"].keys()) >= required, (
            f"cycle {code} missing files: {required - set(cycle['files'])}"
        )


def test_every_file_has_relpath_ending_in_xpt() -> None:
    m = load_manifest()
    for cycle in m["cycles"].values():
        for spec in cycle["files"].values():
            assert spec["relpath"].endswith(".XPT")


def test_base_url_is_cdc() -> None:
    m = load_manifest()
    assert m["base_url"].startswith("https://wwwn.cdc.gov/")


def test_variable_specs_present_for_every_pathogen() -> None:
    m = load_manifest()
    required_pathogens = {"demo", "hcv", "hsv1", "hsv2", "hpv", "ebv"}
    assert set(m["variables"].keys()) == required_pathogens


# ---------- Harmonization helpers ----------

def test_coerce_sero_maps_positive_and_negative() -> None:
    s = pd.Series([1, 2, 3, 1, 2, None])
    out = _coerce_sero(s, positive=[1], negative=[2])
    assert out.dtype.name == "Int8"
    assert out.iloc[0] == 1
    assert out.iloc[1] == 0
    assert pd.isna(out.iloc[2])  # 3 (indeterminate) -> NA
    assert out.iloc[3] == 1
    assert out.iloc[4] == 0
    assert pd.isna(out.iloc[5])  # explicit None -> NA


def test_coerce_sero_preserves_index() -> None:
    s = pd.Series([1, 2, 1], index=[100, 200, 300])
    out = _coerce_sero(s, positive=[1], negative=[2])
    assert out.index.tolist() == [100, 200, 300]


def test_derive_hpv_hr_any_positive_is_positive() -> None:
    hpv = pd.DataFrame(
        {
            "LBXH16": [1, 2, 2, 2, 1],
            "LBXH18": [2, 1, 2, 2, 1],
            "LBXH31": [2, 2, 1, 2, 2],
        }
    )
    spec = {
        "high_risk_type_vars": ["LBXH16", "LBXH18", "LBXH31"],
        "positive_values": [1],
        "negative_values": [2],
    }
    out = _derive_hpv_hr(hpv, spec)
    # rows 0, 1, 2, 4 have at least one positive; row 3 is all-negative
    assert out.iloc[0] == 1
    assert out.iloc[1] == 1
    assert out.iloc[2] == 1
    assert out.iloc[3] == 0
    assert out.iloc[4] == 1


def test_derive_hpv_hr_all_negative_is_negative() -> None:
    hpv = pd.DataFrame({"LBXH16": [2, 2], "LBXH18": [2, 2]})
    spec = {
        "high_risk_type_vars": ["LBXH16", "LBXH18"],
        "positive_values": [1],
        "negative_values": [2],
    }
    out = _derive_hpv_hr(hpv, spec)
    assert (out == 0).all()


def test_derive_hpv_hr_missing_vars_logs_and_returns_na() -> None:
    hpv = pd.DataFrame({"OTHER": [1, 2]})
    spec = {
        "high_risk_type_vars": ["LBXH16", "LBXH18"],
        "positive_values": [1],
        "negative_values": [2],
    }
    out = _derive_hpv_hr(hpv, spec)
    assert out.isna().all()
