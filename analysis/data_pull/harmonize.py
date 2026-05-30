"""Harmonize NHANES per-cycle XPT files into a tidy per-participant table.

Output schema (one row per (seqn, cycle)):
    seqn (int), cycle (str), age (float), sex (int 1=M, 2=F), race (int),
    education (int), poverty_ratio (float),
    weight_mec_2y (float), psu (int), stratum (int),
    sero_hcv  (Int8: 1=positive, 0=negative, NA=missing/indeterminate)
    sero_hsv1 (Int8)
    sero_hsv2 (Int8)
    sero_hpv_hr (Int8 — any high-risk HPV type seropositive)
    sero_ebv  (Int8)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyreadstat

logger = logging.getLogger(__name__)


def _coerce_sero(
    series: pd.Series, positive: list[int], negative: list[int]
) -> pd.Series:
    """Map raw NHANES sero coding to Int8: 1=pos, 0=neg, NA=anything else."""
    out = pd.Series(pd.NA, index=series.index, dtype="Int8")
    out.loc[series.isin(positive)] = 1
    out.loc[series.isin(negative)] = 0
    return out


def _read_xpt(path: Path) -> pd.DataFrame | None:
    """Read a SAS XPT file. Return None if file missing or unreadable."""
    if not path.exists():
        logger.warning("file not found: %s", path)
        return None
    try:
        df, _ = pyreadstat.read_xport(str(path))
        return df
    except Exception as exc:
        logger.warning("failed to read %s: %s", path, exc)
        return None


def _derive_hpv_hr(
    hpv_raw: pd.DataFrame,
    hr_vars: list[str],
    positive: list[int],
    negative: list[int],
) -> pd.Series:
    """Derive 'any high-risk HPV seropositive' from type-specific variables."""
    available = [v for v in hr_vars if v in hpv_raw.columns]
    if not available:
        logger.warning(
            "no requested HPV variables found in columns %s", list(hpv_raw.columns)[:8]
        )
        return pd.Series(pd.NA, index=hpv_raw.index, dtype="Int8")

    any_pos = (hpv_raw[available].isin(positive)).any(axis=1)
    all_neg = (hpv_raw[available].isin(negative)).all(axis=1)
    out = pd.Series(pd.NA, index=hpv_raw.index, dtype="Int8")
    out.loc[all_neg] = 0
    out.loc[any_pos] = 1   # positive trumps negative if any single type is positive
    return out


def harmonize_cycle(
    cycle_code: str, paths: dict[str, Path], spec: dict
) -> pd.DataFrame:
    """Build a tidy per-participant table for a single NHANES cycle."""
    dvar = spec["variables"]["demo"]

    # Demographics — anchor table; every other file left-joins onto this.
    demo = _read_xpt(paths["demo"])
    if demo is None:
        raise RuntimeError(f"DEMO file missing for cycle {cycle_code}; cannot proceed")

    df = pd.DataFrame(
        {
            "seqn": demo[dvar["seqn"]].astype("int64"),
            "cycle": cycle_code,
            "age": demo.get(dvar["age"]),
            "sex": demo.get(dvar["sex"]),
            "race": demo.get(dvar["race"]),
            "education": demo.get(dvar["education"]),
            "poverty_ratio": demo.get(dvar["poverty_ratio"]),
            "weight_mec_2y": demo.get(dvar["weight_mec_2y"]),
            "psu": demo.get(dvar["psu"]),
            "stratum": demo.get(dvar["stratum"]),
        }
    )

    # HCV
    hcv_var = spec["variables"]["hcv"]["sero_var_by_cycle"].get(cycle_code)
    df = _attach_single_var(
        df, paths.get("hcv"), hcv_var,
        spec["variables"]["hcv"]["positive_values"],
        spec["variables"]["hcv"]["negative_values"],
        out_name="sero_hcv",
    )

    # HSV-1 and HSV-2 share one file
    hsv_path = paths.get("hsv")
    for pathogen_key, out_name in [("hsv1", "sero_hsv1"), ("hsv2", "sero_hsv2")]:
        v = spec["variables"][pathogen_key]
        df = _attach_single_var(
            df, hsv_path, v["sero_var"],
            v["positive_values"], v["negative_values"],
            out_name=out_name,
        )

    # EBV (children 6-19; mostly missing for adults). Per-cycle variable name.
    ebv_cfg = spec["variables"]["ebv"]
    ebv_var = ebv_cfg["sero_var_by_cycle"].get(cycle_code)
    df = _attach_single_var(
        df, paths.get("ebv"), ebv_var,
        ebv_cfg["positive_values"], ebv_cfg["negative_values"],
        out_name="sero_ebv",
    )

    # HPV high-risk (multi-variable derivation, per-cycle variable list).
    hpv_cfg = spec["variables"]["hpv"]
    hr_vars = hpv_cfg["high_risk_type_vars_by_cycle"].get(cycle_code, [])
    df = _attach_hpv(
        df, paths.get("hpv"), hr_vars,
        hpv_cfg["positive_values"], hpv_cfg["negative_values"],
    )

    return df


def _attach_single_var(
    df: pd.DataFrame,
    pathogen_path: Path | None,
    sero_var: str | None,
    positive: list[int],
    negative: list[int],
    *,
    out_name: str,
) -> pd.DataFrame:
    if pathogen_path is None or sero_var is None:
        df[out_name] = pd.NA
        return df
    raw = _read_xpt(pathogen_path)
    if raw is None:
        df[out_name] = pd.NA
        return df
    if sero_var not in raw.columns:
        logger.warning(
            "variable %s not in %s; columns sample: %s",
            sero_var, pathogen_path.name, list(raw.columns)[:8],
        )
        df[out_name] = pd.NA
        return df
    coded = _coerce_sero(raw[sero_var], positive=positive, negative=negative)
    sub = pd.DataFrame({"seqn": raw["SEQN"].astype("int64"), out_name: coded})
    return df.merge(sub, on="seqn", how="left")


def _attach_hpv(
    df: pd.DataFrame,
    hpv_path: Path | None,
    hr_vars: list[str],
    positive: list[int],
    negative: list[int],
) -> pd.DataFrame:
    if hpv_path is None or not hr_vars:
        df["sero_hpv_hr"] = pd.NA
        return df
    raw = _read_xpt(hpv_path)
    if raw is None:
        df["sero_hpv_hr"] = pd.NA
        return df
    sero = _derive_hpv_hr(raw, hr_vars, positive, negative)
    sub = pd.DataFrame({"seqn": raw["SEQN"].astype("int64"), "sero_hpv_hr": sero})
    return df.merge(sub, on="seqn", how="left")
