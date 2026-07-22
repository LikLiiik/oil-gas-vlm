from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from geo_adapter.errors import InputDataError


def read_structured_table(path: Path) -> pd.DataFrame:
    """Read CSV, JSON, or YAML records into a DataFrame."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        else:
            raise InputDataError(f"不支持的结构化表格式: {suffix}")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise InputDataError(f"读取 {path} 失败: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("data", [payload]))
    if not isinstance(payload, list):
        raise InputDataError(f"{path} 应包含记录对象或记录数组")
    return pd.DataFrame(payload)


def first_record_as_dict(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise InputDataError("输入表为空")
    record = frame.iloc[0].to_dict()
    return {str(key): (None if pd.isna(value) else value) for key, value in record.items()}

