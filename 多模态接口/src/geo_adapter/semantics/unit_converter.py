from __future__ import annotations

from typing import Any

import numpy as np


def _unit_key(unit: str | None) -> str:
    return (unit or "").strip().lower().replace(" ", "").replace(".", "_")


def convert_curve_unit(
    canonical_name: str, values: np.ndarray, original_unit: str | None
) -> tuple[np.ndarray, str, list[dict[str, Any]], list[str]]:
    """Convert a physical curve to its canonical unit conservatively."""
    result = np.asarray(values, dtype=float).copy()
    unit = _unit_key(original_unit)
    steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    canonical_units = {
        "GR": "API",
        "SP": "mV",
        "CAL": "inch",
        "RES_DEEP": "ohm_m",
        "RES_MEDIUM_SHALLOW": "ohm_m",
        "RES_MICRO": "ohm_m",
        "AC": "us/m",
        "DEN": "g/cm3",
        "CNL": "fraction",
    }
    target = canonical_units[canonical_name]

    if canonical_name == "AC":
        if unit in {"us/ft", "µs/ft", "μs/ft", "usec/ft"}:
            result *= 3.280839895013123
            steps.append({"operation": "unit_conversion", "from": original_unit, "to": target, "factor": 3.280839895013123})
        elif unit in {"us/m", "µs/m", "μs/m", "usec/m"}:
            steps.append({"operation": "unit_verified", "unit": target})
        else:
            warnings.append("AC/DT 单位未确认，数值保留但不得用于可靠时深积分")
    elif canonical_name == "CAL":
        if unit in {"mm", "millimeter", "millimetre"}:
            result /= 25.4
            steps.append({"operation": "unit_conversion", "from": original_unit, "to": target, "factor": 1 / 25.4})
        elif unit not in {"in", "inch", "inches", ""}:
            warnings.append(f"CAL 单位 {original_unit!r} 未识别")
    elif canonical_name.startswith("RES_"):
        recognized = {"ohm_m", "ohmm", "ohm-m", "ohm·m", "ohm/m"}
        if unit and unit not in recognized:
            warnings.append(f"电阻率单位 {original_unit!r} 未转换，请核实是否为 ohm_m")
    elif canonical_name == "DEN":
        if unit in {"kg/m3", "kg/m^3", "kgm-3"}:
            result /= 1000.0
            steps.append({"operation": "unit_conversion", "from": original_unit, "to": target, "factor": 0.001})
        elif unit and unit not in {"g/cm3", "g/cm^3", "g/c3", "g/cc", "gcc"}:
            warnings.append(f"密度单位 {original_unit!r} 未转换")
    elif canonical_name == "CNL":
        if unit in {"%", "percent", "percentage", "pu", "p_u"}:
            result /= 100.0
            steps.append({"operation": "unit_conversion", "from": original_unit, "to": target, "factor": 0.01})
        elif unit in {"v/v", "fraction", "decimal", "dec", ""}:
            if not unit:
                finite = result[np.isfinite(result)]
                if finite.size and np.nanpercentile(np.abs(finite), 95) > 1.5:
                    warnings.append("CNL/NPHI 缺少单位且数值可能为百分数；未静默除以 100")
                else:
                    warnings.append("CNL/NPHI 缺少单位；按原值保留，单位判断置信度低")
        else:
            warnings.append(f"CNL/NPHI 单位 {original_unit!r} 含糊，未转换")
    elif canonical_name == "GR" and unit and unit not in {"api", "gapi"}:
        warnings.append(f"GR 单位 {original_unit!r} 未转换")
    elif canonical_name == "SP" and unit in {"v", "volt", "volts"}:
        result *= 1000.0
        steps.append({"operation": "unit_conversion", "from": original_unit, "to": target, "factor": 1000.0})
    return result, target, steps, warnings
