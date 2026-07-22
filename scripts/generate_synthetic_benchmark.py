"""Generate labeled synthetic geo-adapter packages for inference benchmarking.

The generated geology is deliberately simple and is suitable for software,
robustness, and regression tests only.  It is not a replacement for a public
or field geological benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--shape", default="32,48,192", help="inline,xline,sample")
    parser.add_argument("--workspace", default="synthetic_benchmark")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ricker(length: int = 25, frequency: float = 0.18) -> np.ndarray:
    x = np.arange(length, dtype=np.float32) - (length - 1) / 2
    a = (np.pi * frequency * x) ** 2
    wavelet = (1.0 - 2.0 * a) * np.exp(-a)
    return wavelet.astype(np.float32)


def make_seismic(seed: int, shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    n_il, n_xl, n_s = shape
    reflectivity = np.zeros(shape, dtype=np.float32)
    fault_mask = np.zeros(shape, dtype=np.uint8)
    channel_mask = np.zeros(shape, dtype=np.uint8)
    reservoir_mask = np.zeros(shape, dtype=np.uint8)

    throw = int(rng.integers(5, 13))
    fault_base = float(rng.uniform(0.35, 0.65) * n_xl)
    fault_slope = float(rng.uniform(-0.20, 0.20))
    layers = np.linspace(0.16, 0.86, 8) * n_s
    layer_amp = rng.uniform(-1.0, 1.0, len(layers))
    layer_amp[np.abs(layer_amp) < 0.25] += 0.45

    channel_depth = float(rng.uniform(0.43, 0.68) * n_s)
    channel_width = float(rng.uniform(3.5, 7.0))
    channel_thickness = int(rng.integers(3, 7))

    for il in range(n_il):
        fault_x = fault_base + fault_slope * (il - n_il / 2)
        fx = int(np.clip(round(fault_x), 1, n_xl - 2))
        fault_mask[il, max(0, fx - 1):min(n_xl, fx + 2), 8:n_s - 8] = 1
        channel_center = n_xl * 0.52 + 0.18 * n_xl * np.sin(il / max(n_il - 1, 1) * 2 * np.pi)

        for xl in range(n_xl):
            displacement = throw if xl >= fault_x else 0
            undulation = 2.2 * np.sin(il / 7.0 + xl / 11.0)
            for base, amplitude in zip(layers, layer_amp):
                sample = int(round(base + undulation + displacement))
                if 1 <= sample < n_s - 1:
                    reflectivity[il, xl, sample] += float(amplitude)

            lateral = (xl - channel_center) / channel_width
            if abs(lateral) <= 1.0:
                center = int(round(channel_depth + 3.0 * np.sin(il / 5.0) + displacement))
                half = max(1, int(round(channel_thickness * np.sqrt(max(0.0, 1.0 - lateral * lateral)))))
                top, bottom = max(1, center - half), min(n_s - 2, center + half)
                channel_mask[il, xl, top:bottom + 1] = 1
                reflectivity[il, xl, top] -= 1.4
                reflectivity[il, xl, bottom] += 1.2
                if abs(lateral) <= 0.65:
                    reservoir_mask[il, xl, top:bottom + 1] = 1

    wavelet = ricker()
    seismic = np.empty_like(reflectivity)
    for il in range(n_il):
        for xl in range(n_xl):
            seismic[il, xl] = np.convolve(reflectivity[il, xl], wavelet, mode="same")

    noise_sigma = float(rng.uniform(0.04, 0.18))
    gain = np.linspace(0.8, 1.25, n_s, dtype=np.float32)
    seismic = seismic * gain[None, None, :]
    seismic += rng.normal(0.0, noise_sigma, shape).astype(np.float32)
    seismic /= max(float(np.std(seismic)), 1e-6)

    labels = {
        "fault": fault_mask,
        "channel": channel_mask,
        "reservoir_candidate": reservoir_mask,
    }
    metadata = {
        "seed": seed,
        "shape": list(shape),
        "fault_throw_samples": throw,
        "fault_base_xline_index": fault_base,
        "fault_slope": fault_slope,
        "channel_depth_sample": channel_depth,
        "noise_sigma": noise_sigma,
    }
    return seismic.astype(np.float32), labels, metadata


def make_well(seed: int, path: Path, missing_mode: int) -> None:
    rng = np.random.default_rng(seed + 991)
    depth = np.arange(1000.0, 1200.5, 0.5)
    phase = (depth - depth.min()) / 18.0
    sand = ((depth >= 1070) & (depth <= 1100)) | ((depth >= 1140) & (depth <= 1175))
    gas = (depth >= 1148) & (depth <= 1168)
    frame = pd.DataFrame({
        "MD": depth,
        "GR": np.where(sand, 38.0, 92.0) + rng.normal(0, 4, len(depth)),
        "SP": np.where(sand, -32.0, -7.0) + rng.normal(0, 2, len(depth)),
        "CALI": 8.5 + rng.normal(0, 0.08, len(depth)),
        "ILD": np.where(gas, 70.0, np.where(sand, 14.0, 5.0)) * np.exp(rng.normal(0, 0.08, len(depth))),
        "LLD": np.where(gas, 65.0, np.where(sand, 13.0, 5.0)) * np.exp(rng.normal(0, 0.08, len(depth))),
        "ILM": np.where(gas, 32.0, np.where(sand, 9.0, 4.0)) * np.exp(rng.normal(0, 0.08, len(depth))),
        "MSFL": np.where(sand, 5.5, 3.0) * np.exp(rng.normal(0, 0.06, len(depth))),
        "DTC": np.where(sand, 245.0, 315.0) + rng.normal(0, 5, len(depth)),
        "RHOB": np.where(gas, 2.18, np.where(sand, 2.30, 2.52)) + rng.normal(0, 0.025, len(depth)),
        "NPHI": np.where(gas, 0.11, np.where(sand, 0.20, 0.33)) + rng.normal(0, 0.015, len(depth)),
    })
    # Deterministic industrial missing-data cases, without inventing replacement curves.
    if missing_mode == 1:
        frame.loc[80:86, "DTC"] = np.nan
    elif missing_mode == 2:
        frame["MSFL"] = np.nan
    elif missing_mode == 3:
        frame.loc[180:230, ["ILD", "LLD"]] = np.nan
    frame.to_csv(path, index=False, encoding="utf-8")


def write_supporting_tables(raw_dir: Path, seed: int) -> None:
    well = f"SYN_{seed}"
    pd.DataFrame([{
        "WELL": well, "X": 500000.0, "Y": 6500000.0,
        "KB": 75.0, "GL": 69.0, "TD": 1200.0, "CRS": "EPSG:32631",
    }]).to_csv(raw_dir / "well_location.csv", index=False)
    md = np.arange(1000.0, 1201.0, 10.0)
    pd.DataFrame({
        "WELL": well, "MD": md, "TVD": 1000.0 + 0.985 * (md - 1000.0),
        "DX": 0.10 * (md - 1000.0), "DY": 0.06 * (md - 1000.0),
    }).to_csv(raw_dir / "trajectory.csv", index=False)
    pd.DataFrame({
        "TVDSS": [925.0, 970.0, 1020.0, 1080.0, 1120.0],
        "TWT_MS": [600.0, 628.0, 660.0, 699.0, 725.0],
    }).to_csv(raw_dir / "control_points.csv", index=False)


def build_config(project: Path, workspace: Path, sample_id: str, raw_dir: Path, shape: tuple[int, int, int]) -> Path:
    adapter_root = project / "多模态接口"
    template = yaml.safe_load((adapter_root / "examples/sample_config.yaml").read_text(encoding="utf-8"))
    template["sample_id"] = sample_id
    template["inputs"]["seismic"]["path"] = str((raw_dir / "seismic.npz").resolve())
    template["inputs"]["seismic"]["domain"] = "time"
    template["inputs"]["well_log"]["path"] = str((raw_dir / "well.csv").resolve())
    template["inputs"]["well_log"]["well_id"] = sample_id.upper()
    template["inputs"]["well_location"]["path"] = str((raw_dir / "well_location.csv").resolve())
    template["inputs"]["trajectory"]["path"] = str((raw_dir / "trajectory.csv").resolve())
    template["processing"]["time_depth"]["calibration"]["control_points_path"] = str((raw_dir / "control_points.csv").resolve())
    template["processing"]["seismic"].update({
        "inline_index": shape[0] // 2,
        "crossline_index": shape[1] // 2,
        "sample_index": shape[2] // 2,
        "local_patch_radius": min(12, shape[0] // 3, shape[1] // 3),
    })
    template["curve_aliases_path"] = str((adapter_root / "configs/curve_aliases.yaml").resolve())
    template["field_aliases_path"] = str((adapter_root / "configs/field_aliases.yaml").resolve())
    template["prompt_templates_path"] = str((adapter_root / "configs/prompt_templates.yaml").resolve())
    template["output"] = {
        "directory": str((workspace / "runs" / sample_id).resolve()),
        "overwrite": True,
    }
    config_dir = workspace / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{sample_id}.yaml"
    config_path.write_text(yaml.safe_dump(template, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def main() -> int:
    args = parse_args()
    shape_parts = tuple(int(value) for value in args.shape.split(","))
    if len(shape_parts) != 3 or min(shape_parts) < 8:
        raise SystemExit("--shape must contain three dimensions >= 8")
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be positive")

    project = Path(__file__).resolve().parents[1]
    workspace = (project / args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    from geo_adapter.pipeline import prepare_geo_sample

    index: list[dict] = []
    for offset in range(args.num_samples):
        seed = args.start_seed + offset
        sample_id = f"synthetic_{seed:06d}"
        run_dir = workspace / "runs" / sample_id
        if run_dir.exists() and not args.overwrite:
            index.append({
                "sample_id": sample_id,
                "seed": seed,
                "status": "exists",
                "run_dir": str(run_dir),
                "label_dir": str(workspace / "labels" / sample_id),
                "warnings": [],
                "errors": [],
            })
            print(f"[skip] {sample_id}: already exists")
            continue

        raw_dir = workspace / "raw" / sample_id
        label_dir = workspace / "labels" / sample_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        seismic, labels, metadata = make_seismic(seed, shape_parts)
        np.savez_compressed(
            raw_dir / "seismic.npz", amplitude=seismic, domain=np.array("time"),
            inline_numbers=np.arange(1000, 1000 + shape_parts[0]),
            crossline_numbers=np.arange(2000, 2000 + shape_parts[1]),
            sample_interval_ms=np.array(2.0),
        )
        for name, mask in labels.items():
            np.save(label_dir / f"{name}_mask.npy", mask, allow_pickle=False)
        (label_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        make_well(seed, raw_dir / "well.csv", missing_mode=offset % 4)
        write_supporting_tables(raw_dir, seed)
        config_path = build_config(project, workspace, sample_id, raw_dir, shape_parts)
        result = prepare_geo_sample(config_path)
        status = "ok" if result.success else "failed"
        entry = {
            "sample_id": sample_id, "seed": seed, "status": status,
            "run_dir": str(run_dir), "label_dir": str(label_dir),
            "warnings": list(result.warnings), "errors": list(result.errors),
        }
        index.append(entry)
        print(f"[{status}] {sample_id}: warnings={len(result.warnings)} errors={len(result.errors)}")

    index_path = workspace / "dataset_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(item["status"] in {"ok", "exists"} for item in index)
    print(f"generated/prepared: {ok}/{len(index)}")
    print(f"index: {index_path}")
    return 0 if ok == len(index) else 1


if __name__ == "__main__":
    raise SystemExit(main())
