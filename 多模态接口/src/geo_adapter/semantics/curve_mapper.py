from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from geo_adapter.models import ProcessedCurve
from geo_adapter.preprocess.missing_data import interpolate_short_gaps, max_consecutive_false
from geo_adapter.schemas.config import WellLogProcessingConfig
from geo_adapter.semantics.unit_converter import convert_curve_unit


CANONICAL_SLOTS = [
    "GR", "SP", "CAL", "RES_DEEP", "RES_MEDIUM_SHALLOW", "RES_MICRO", "AC", "DEN", "CNL"
]

PHYSICAL_QUANTITIES = {
    "GR": "natural_gamma_ray",
    "SP": "spontaneous_potential",
    "CAL": "borehole_caliper",
    "RES_DEEP": "deep_resistivity",
    "RES_MEDIUM_SHALLOW": "medium_or_shallow_resistivity",
    "RES_MICRO": "micro_resistivity",
    "AC": "acoustic_slowness",
    "DEN": "bulk_density",
    "CNL": "neutron_porosity",
}


def _norm(value: str) -> str:
    return "".join(ch for ch in str(value).strip().upper() if ch not in " _-./\\()[]")


def load_curve_aliases(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"曲线别名配置不存在: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _classify_resistivity(name: str, description: str, override: dict[str, str] | None) -> tuple[str, str]:
    if override:
        return override.get("investigation_depth", "unknown"), override.get("measurement_family", "unknown")
    token = _norm(f"{name} {description}")
    family = "unknown"
    if any(key in token for key in ("ILD", "ILM", "INDUCTION", "感应")):
        family = "induction"
    elif any(key in token for key in ("LLD", "LLS", "LATEROLOG", "侧向")):
        family = "laterolog"
    elif any(key in token for key in ("MSFL", "MLL", "MICRO", "微球", "微电")):
        family = "micro_focused"

    if any(key in token for key in ("MSFL", "MLL", "MICRO", "微球", "微电")):
        depth = "micro"
    elif any(key in token for key in ("LLD", "ILD", "RT90", "DEEP", "深")):
        depth = "deep"
    elif any(key in token for key in ("ILM", "LLS", "RT30", "MEDIUM", "SHALLOW", "中", "浅")):
        depth = "medium" if any(key in token for key in ("ILM", "RT30", "MEDIUM", "中")) else "shallow"
    else:
        depth = "unknown"
    return depth, family


def map_and_process_curves(
    frame: pd.DataFrame,
    units: dict[str, str | None],
    descriptions: dict[str, str],
    alias_config: dict[str, Any],
    processing: WellLogProcessingConfig,
) -> dict[str, ProcessedCurve]:
    """Map source columns to fixed independent physical slots and preprocess them."""
    candidates: dict[str, list[tuple[str, int]]] = {slot: [] for slot in CANONICAL_SLOTS}
    for column in frame.columns:
        numeric_candidate = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(numeric_candidate.to_numpy(dtype=float)).any():
            continue
        source_norm = _norm(column)
        description_norm = _norm(descriptions.get(column, ""))
        for slot in CANONICAL_SLOTS:
            aliases = alias_config.get(slot, {}).get("aliases", [])
            match_set = {_norm(slot), *(_norm(alias) for alias in aliases)}
            score = 100 if source_norm in match_set else 0
            if not score and description_norm in match_set:
                score = 60
            # Resistivity classification is physical metadata, not loose name
            # matching. It may route well-known exact mnemonics to a slot.
            if slot.startswith("RES_"):
                depth, _ = _classify_resistivity(
                    str(column), descriptions.get(column, ""), processing.resistivity_overrides.get(str(column))
                )
                routed = {
                    "deep": "RES_DEEP",
                    "medium": "RES_MEDIUM_SHALLOW",
                    "shallow": "RES_MEDIUM_SHALLOW",
                    "micro": "RES_MICRO",
                }.get(depth)
                if routed == slot:
                    score = max(score, 80)
            if score:
                candidates[slot].append((str(column), score))

    mapped: dict[str, ProcessedCurve] = {}
    for slot in CANONICAL_SLOTS:
        slot_candidates = candidates[slot]
        preferred = processing.preferred_curves.get(slot)
        if preferred and preferred in frame.columns:
            if preferred not in [name for name, _ in slot_candidates]:
                slot_candidates.append((preferred, 200))
            else:
                slot_candidates = [(name, 200 if name == preferred else score) for name, score in slot_candidates]

        # Do not let the same resistivity curve populate more than its classified slot.
        def rank(item: tuple[str, int]) -> tuple[int, float, int]:
            name, score = item
            series = pd.to_numeric(frame[name], errors="coerce")
            missing = float((~np.isfinite(series.to_numpy(dtype=float))).mean())
            return score, -missing, int(series.notna().sum())

        slot_candidates.sort(key=rank, reverse=True)
        if not slot_candidates:
            mapped[slot] = ProcessedCurve(
                canonical_name=slot,
                physical_quantity=PHYSICAL_QUANTITIES[slot],
                available=False,
                limitations=["source_curve_missing"],
            )
            continue

        selected, selected_score = slot_candidates[0]
        source_series = frame[selected]
        numeric_series = pd.to_numeric(source_series, errors="coerce")
        raw = numeric_series.to_numpy(dtype=float)
        parse_failure_count = int((source_series.notna() & numeric_series.isna()).sum())
        converted, canonical_unit, steps, conversion_warnings = convert_curve_unit(
            slot, raw, units.get(selected)
        )
        if parse_failure_count:
            steps.append({"operation": "invalidate_non_numeric_tokens", "count": parse_failure_count})
            conversion_warnings.append(f"{parse_failure_count} 个非数值标记已作为缺失处理")
        if slot.startswith("RES_"):
            nonpositive = np.isfinite(converted) & (converted <= 0)
            if nonpositive.any():
                converted[nonpositive] = np.nan
                steps.append({"operation": "invalidate_nonpositive_resistivity", "count": int(nonpositive.sum())})
                conversion_warnings.append(f"{int(nonpositive.sum())} 个非正电阻率值已标记为无效")
        cleaned, valid, interpolated = interpolate_short_gaps(
            converted,
            processing.short_gap_interpolation.max_gap_samples,
            processing.short_gap_interpolation.enabled,
        )
        if interpolated.any():
            steps.append(
                {
                    "operation": "short_gap_interpolation",
                    "method": processing.short_gap_interpolation.method,
                    "max_gap_samples": processing.short_gap_interpolation.max_gap_samples,
                    "count": int(interpolated.sum()),
                }
            )
        depth, family = (None, None)
        if slot.startswith("RES_"):
            depth, family = _classify_resistivity(
                selected, descriptions.get(selected, ""), processing.resistivity_overrides.get(selected)
            )
            steps.append({"operation": "log10_view", "nonpositive_kept_invalid": True})
        alternative_details: list[dict[str, Any]] = []
        for alternative, _ in slot_candidates[1:]:
            alt_depth, alt_family = (None, None)
            if slot.startswith("RES_"):
                alt_depth, alt_family = _classify_resistivity(
                    alternative,
                    descriptions.get(alternative, ""),
                    processing.resistivity_overrides.get(alternative),
                )
            alt_values = pd.to_numeric(frame[alternative], errors="coerce").to_numpy(dtype=float)
            alternative_details.append(
                {
                    "original_mnemonic": alternative,
                    "original_unit": units.get(alternative),
                    "missing_ratio": float((~np.isfinite(alt_values)).mean()),
                    "investigation_depth": alt_depth,
                    "measurement_family": alt_family,
                    "selection_status": "not_averaged",
                }
            )
        mapped[slot] = ProcessedCurve(
            canonical_name=slot,
            physical_quantity=PHYSICAL_QUANTITIES[slot],
            available=bool(valid.any() or interpolated.any()),
            selected_curve=selected,
            alternative_curves=[name for name, _ in slot_candidates[1:]],
            alternative_curve_details=alternative_details,
            original_unit=units.get(selected),
            canonical_unit=canonical_unit,
            mapping_confidence="high" if selected_score >= 100 else "medium",
            selection_reason=(
                "用户配置指定首选曲线" if preferred == selected else "按精确别名、物理分类、缺失率和覆盖长度排序"
            ),
            investigation_depth=depth,
            measurement_family=family,
            raw_values=raw,
            values=cleaned,
            valid_mask=valid,
            interpolated_mask=interpolated,
            missing_ratio=float((~valid).mean()),
            max_gap_samples=max_consecutive_false(valid),
            preprocessing=steps,
            warnings=conversion_warnings,
        )
    return mapped
