"""CLI 入口。

赛题主流程（geo_adapter 前置 → 本 pipeline）:

  # 1) 用 geo_adapter 把原始 sgy/las/csv 处理成标准 run 包
  geo-adapter prepare --config path/to/sample.yaml
  # → 产生 runs/<sample_id>/{manifest.json, prompts/, assets/, schemas/, ...}

  # 2) 用本 pipeline 消费 run 包，出 JSON + 标注 PNG + 属性 SEG-Y
  CUDA_VISIBLE_DEVICES=1 python -m pipeline \\
      --run-dir path/to/runs/<sample_id> \\
      --output-dir out/

  # 关闭 VLM 二次验证（快一半，精度稍降）
  python -m pipeline --run-dir runs/sample --no-verify

fallback / 老接口:
  # 直接从 SEG-Y（跳过 geo_adapter，走内置切片渲染）
  python -m pipeline --input path/to/volume.sgy --tasks fault,horizon --output-dir out/

  # 4-Agent 语义解释流水线（非赛题）
  python -m pipeline --agent all --seismic-image ... --log-image ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _cmd_geological_tasks(args) -> int:
    """--input {segy|image} --tasks fault,...   赛题主流程。"""
    from pipeline import Pipeline
    from pipeline.io import read_segy
    from pipeline.io.geometry import SliceGeometry

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Pipeline(verbose=not args.quiet)

    inp = Path(args.input)
    if inp.suffix.lower() in (".sgy", ".segy"):
        print(f"[cli] SEG-Y input: {inp}")
        vol = read_segy(str(inp), strict=False)
        print(f"[cli] volume shape={vol.cube.shape}, "
              f"dt={vol.sample_interval_ms}ms")
        report = p.run_volume(
            vol, tasks=tasks,
            slice_axis=args.slice_axis,
            slice_stride=args.slice_stride,
            out_dir=out_dir,
        )
    else:
        print(f"[cli] 2D image input: {inp}")
        img = _load_image(str(inp))
        # 用户没给 geometry：用像素坐标作为数据坐标（bbox 仍可视化）
        geom = SliceGeometry(
            axis_x_name="pixel_x", axis_y_name="pixel_y",
            x_min=0, x_max=img.width, y_top=0, y_bottom=img.height,
            pixel_width=img.width, pixel_height=img.height,
            slice_kind="single", slice_index=None,
        )
        report = p.run_slice_for_tasks(img, geom, tasks, out_dir=out_dir)
        # 单切片报告结构化落盘
        from pipeline.exporter import summary_report
        summary_report({"single_slice": {t: v["result"].to_dict()
                                          for t, v in report.items()}},
                       out_dir)

    print(f"\n[cli] wrote outputs to {out_dir}/")
    print(f"[cli] see {out_dir}/report.json for the summary")
    return 0


def _cmd_agents(args) -> int:
    """--agent all|seismic|log|fusion  内部 4-Agent 流水线。"""
    from pipeline import Pipeline

    p = Pipeline(verbose=not args.quiet)
    td_pairs = json.loads(args.time_depth) if args.time_depth else None
    well_info = json.loads(args.well_info) if args.well_info else None

    if args.agent == "all":
        out = p.run_all(
            seismic_image=_load_image(args.seismic_image) if args.seismic_image else None,
            log_image=_load_image(args.log_image) if args.log_image else None,
            fusion_image=_load_image(args.fusion_image) if args.fusion_image else None,
            time_depth_pairs=td_pairs, well_info=well_info,
            prospect_image=_load_image(args.prospect_image) if args.prospect_image else None,
        )
        payload = out.to_dict()
    else:
        img_path = args.image or {
            "seismic": args.seismic_image, "log": args.log_image,
            "fusion":  args.fusion_image,
        }.get(args.agent)
        if img_path is None and args.agent != "prospect":
            print(f"缺少图像。为 --agent {args.agent} 提供 --image 或 "
                  f"--{args.agent}-image", file=sys.stderr)
            return 2
        if args.agent == "seismic":
            r = p.run_seismic(_load_image(img_path))
        elif args.agent == "log":
            r = p.run_log(_load_image(img_path))
        elif args.agent == "fusion":
            r = p.run_fusion(_load_image(img_path), td_pairs, well_info)
        elif args.agent == "prospect":
            print("--agent prospect 需要前置产物，请用 --agent all", file=sys.stderr)
            return 2
        else:
            print(f"unknown agent {args.agent}", file=sys.stderr); return 2
        payload = r.to_dict()

    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[cli] wrote {args.output}")
    return 0


def _cmd_adapter(args) -> int:
    """--run-dir runs/<sample>   赛题主流程（走 geo_adapter）。"""
    from pipeline import Pipeline

    p = Pipeline(verbose=not args.quiet)
    report = p.run_from_adapter(
        run_dir=args.run_dir,
        out_dir=args.output_dir,
        verify=not args.no_verify,
        max_iterations=args.max_iter,
    )
    print(f"[cli] sample_id={report.get('sample_id')} ok={report.get('ok')}")
    return 0 if report.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Oil-Gas VLM Agent Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 赛题主流程（geo_adapter 前置）
    p.add_argument("--run-dir", type=str,
                   help="geo_adapter 产出的 run 目录，形如 runs/<sample_id>/")
    p.add_argument("--no-verify", action="store_true",
                   help="关闭 VLM 二次验证回环（快一半，精度稍降）")
    p.add_argument("--max-iter", type=int, default=3,
                   help="VLM 验证回环最大迭代次数，默认 3")
    # Fallback：直接读 SEG-Y
    p.add_argument("--input", type=str,
                   help="输入文件：SEG-Y (.sgy/.segy) 或 2D 图像 (.png/.jpg)")
    p.add_argument("--tasks", type=str, default="fault,horizon,facies,fracture",
                   help="逗号分隔的任务列表（fault/horizon/facies/fracture）。"
                        "默认 4 类全跑")
    p.add_argument("--slice-axis", choices=["inline", "crossline"],
                   default="inline",
                   help="SEG-Y 切片方向，默认 inline")
    p.add_argument("--slice-stride", type=int, default=5,
                   help="每 N 个索引取一张切片，默认 5")
    p.add_argument("--output-dir", type=str, default="out",
                   help="输出目录，默认 ./out")

    # 内部 4-Agent 流水线模式（老接口，非赛题）
    p.add_argument("--agent",
                   choices=["seismic", "log", "fusion", "prospect", "all"],
                   help="启用 4-Agent 流水线模式（与 --input 二选一）")
    p.add_argument("--seismic-image", type=str)
    p.add_argument("--log-image",     type=str)
    p.add_argument("--fusion-image",  type=str)
    p.add_argument("--prospect-image", type=str)
    p.add_argument("--image", type=str, help="单 agent 模式的通用图像路径")
    p.add_argument("--time-depth", type=str)
    p.add_argument("--well-info", type=str)
    p.add_argument("--output", type=str, default="/tmp/pipeline_output.json",
                   help="4-Agent 模式输出 JSON 路径")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.run_dir:
        return _cmd_adapter(args)
    if args.input:
        return _cmd_geological_tasks(args)
    if args.agent:
        return _cmd_agents(args)

    print("需要 --run-dir（赛题主流程）、--input（fallback SEG-Y）"
          "或 --agent（4-Agent 语义解释）之一。使用 -h 查看示例。",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
