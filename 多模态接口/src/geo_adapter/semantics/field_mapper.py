from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _norm(value: str) -> str:
    return "".join(ch for ch in str(value).strip().upper() if ch not in " _-./\\()[]")


def map_fields(frame: pd.DataFrame, aliases: dict[str, Iterable[str]]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename exact normalized aliases to canonical structured-data fields."""
    normalized = {_norm(column): str(column) for column in frame.columns}
    rename: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in [canonical, *list(candidates)]:
            source = normalized.get(_norm(candidate))
            if source is not None:
                rename[source] = canonical
                mapping[canonical] = source
                break
    return frame.rename(columns=rename), mapping

