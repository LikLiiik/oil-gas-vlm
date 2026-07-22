"""Lightweight smoke-check for SAM-family mask outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="SAM mask JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, p1 in enumerate(points):
        p2 = points[(i + 1) % len(points)]
        area += p1[0] * p2[1] - p2[0] * p1[1]
    return abs(area) / 2.0


def main() -> int:
    payload = load_json(Path(parse_args().input).expanduser().resolve())
    records = payload.get("records", [])
    total_masks = 0
    total_area = 0.0
    samples = []
    for record in records:
        masks = record.get("masks", []) or []
        areas = []
        for mask in masks:
            polygon = mask.get("polygon_xy") if isinstance(mask, dict) else None
            if isinstance(polygon, list):
                area = polygon_area(polygon)
                areas.append(area)
                total_area += area
        total_masks += len(masks)
        samples.append(
            {
                "sample_id": record.get("sample_id"),
                "masks": len(masks),
                "total_polygon_area": round(sum(areas), 2),
                "labels": sorted({mask.get("label") for mask in masks if isinstance(mask, dict)}),
            }
        )
    summary = {
        "records": len(records),
        "total_masks": total_masks,
        "total_polygon_area": round(total_area, 2),
        "samples": samples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
