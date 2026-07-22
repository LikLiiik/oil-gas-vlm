"""Generate tiny deterministic synthetic inputs for software-flow testing only.

These values do not represent a real field or reliable geological laws.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260720)

    inline = np.arange(24)[:, None, None]
    crossline = np.arange(28)[None, :, None]
    sample = np.arange(96)[None, None, :]
    seismic = (
        np.sin(sample / 7.0 + inline / 8.0)
        + 0.35 * np.cos(sample / 11.0 + crossline / 5.0)
        + 0.08 * rng.normal(size=(24, 28, 96))
    ).astype(np.float32)
    np.savez_compressed(
        data_dir / "demo_seismic.npz",
        amplitude=seismic,
        domain=np.array("time"),
        inline_numbers=np.arange(100, 124),
        crossline_numbers=np.arange(200, 228),
        sample_interval_ms=np.array(2.0),
    )

    depth = np.arange(1000.0, 1100.5, 0.5)
    phase = (depth - depth.min()) / 12.0
    ac = 300.0 + 12.0 * np.sin(phase / 1.7)
    ac[70:72] = np.nan  # bounded short gap, interpolated by default config
    well = pd.DataFrame(
        {
            "MD": depth,
            "GR": 70.0 + 22.0 * np.sin(phase) + rng.normal(0, 2, len(depth)),
            "SP": -20.0 + 9.0 * np.cos(phase / 1.3),
            "CALI": 8.5 + 0.18 * np.sin(phase / 2.0),
            "ILD": 18.0 * np.exp(0.3 * np.sin(phase)),
            "LLD": 17.0 * np.exp(0.28 * np.sin(phase + 0.2)),
            "ILM": 9.0 * np.exp(0.22 * np.cos(phase)),
            "MSFL": 4.0 * np.exp(0.15 * np.sin(phase * 1.2)),
            "DTC": ac,
            "RHOB": 2.35 + 0.08 * np.cos(phase / 1.4),
            "NPHI": 0.22 + 0.04 * np.sin(phase / 1.8),
        }
    )
    well.to_csv(data_dir / "demo_well.csv", index=False, encoding="utf-8")

    pd.DataFrame(
        [
            {
                "WELL": "DEMO_WELL",
                "X": 500000.0,
                "Y": 6500000.0,
                "KB": 75.0,
                "GL": 69.0,
                "TD": 1100.0,
                "CRS": "EPSG:32631",
            }
        ]
    ).to_csv(data_dir / "demo_well_location.csv", index=False, encoding="utf-8")

    md = np.arange(1000.0, 1101.0, 10.0)
    tvd = 1000.0 + 0.98 * (md - 1000.0)
    x_offset = 0.12 * (md - 1000.0)
    y_offset = 0.08 * (md - 1000.0)
    pd.DataFrame(
        {
            "WELL": "DEMO_WELL",
            "MD": md,
            "TVD": tvd,
            "DX": x_offset,
            "DY": y_offset,
        }
    ).to_csv(data_dir / "demo_trajectory.csv", index=False, encoding="utf-8")

    pd.DataFrame(
        {
            "TVDSS": [925.0, 950.0, 975.0, 1000.0],
            "TWT_MS": [600.0, 615.4, 630.2, 645.5],
        }
    ).to_csv(data_dir / "demo_control_points.csv", index=False, encoding="utf-8")
    (data_dir / "SYNTHETIC_DATA_NOTICE.txt").write_text(
        "仅用于测试软件流程，不代表真实地质规律。\n不包含真实井、真实坐标或真实油气藏标签。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

