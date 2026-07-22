from __future__ import annotations

import pandas as pd

from geo_adapter.schemas.config import WellLogProcessingConfig
from geo_adapter.semantics.curve_mapper import CANONICAL_SLOTS, load_curve_aliases, map_and_process_curves


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GR": [60.0, 70.0, 80.0],
            "SP": [-10.0, -8.0, -9.0],
            "CALI": [8.5, 8.6, 8.4],
            "ILD": [10.0, 20.0, 30.0],
            "LLD": [11.0, 21.0, 31.0],
            "ILM": [5.0, 6.0, 7.0],
            "MSFL": [2.0, 3.0, 4.0],
            "DTC": [300.0, 301.0, 302.0],
            "RHOB": [2.3, 2.4, 2.5],
            "NPHI": [0.2, 0.22, 0.24],
        }
    )


def test_complete_nine_slots_and_resistivity_families(project_root) -> None:
    aliases = load_curve_aliases(project_root / "configs/curve_aliases.yaml")
    processing = WellLogProcessingConfig(
        preferred_curves={"RES_DEEP": "ILD"},
        curve_units={
            "GR": "API", "SP": "mV", "CALI": "inch", "ILD": "ohm_m", "LLD": "ohm_m",
            "ILM": "ohm_m", "MSFL": "ohm_m", "DTC": "us/m", "RHOB": "g/cm3", "NPHI": "fraction",
        },
        curve_descriptions={"ILD": "deep induction", "LLD": "deep laterolog", "ILM": "medium induction", "MSFL": "micro focused"},
    )
    curves = map_and_process_curves(_frame(), processing.curve_units, processing.curve_descriptions, aliases, processing)
    assert list(curves) == CANONICAL_SLOTS
    assert all(curve.available for curve in curves.values())
    assert curves["RES_DEEP"].selected_curve == "ILD"
    assert "LLD" in curves["RES_DEEP"].alternative_curves
    assert curves["RES_DEEP"].measurement_family == "induction"
    assert curves["RES_DEEP"].alternative_curve_details[0]["measurement_family"] == "laterolog"
    assert curves["RES_MEDIUM_SHALLOW"].selected_curve == "ILM"
    assert curves["RES_MICRO"].selected_curve == "MSFL"


def test_whole_sp_missing_is_not_substituted(project_root) -> None:
    frame = _frame().drop(columns=["SP"])
    aliases = load_curve_aliases(project_root / "configs/curve_aliases.yaml")
    curves = map_and_process_curves(frame, {}, {}, aliases, WellLogProcessingConfig())
    assert not curves["SP"].available
    assert curves["SP"].selected_curve is None
    assert curves["GR"].selected_curve == "GR"


def test_nonpositive_resistivity_is_invalidated(project_root) -> None:
    frame = pd.DataFrame({"ILD": [10.0, 0.0, -1.0, 20.0]})
    aliases = load_curve_aliases(project_root / "configs/curve_aliases.yaml")
    curves = map_and_process_curves(frame, {"ILD": "ohm_m"}, {"ILD": "deep induction"}, aliases, WellLogProcessingConfig())
    curve = curves["RES_DEEP"]
    assert curve.valid_mask.tolist() == [True, False, False, True]
    assert any(step["operation"] == "invalidate_nonpositive_resistivity" for step in curve.preprocessing)
