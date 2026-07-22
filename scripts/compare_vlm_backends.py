"""A/B 对比两个 VLM 后端：本地 Qwen vs OpenAI 兼容 API。

默认 **不** 自动真实调用——只把两边要跑的 pipeline 准备好，参数差异打印出来。
要真正跑必须显式传 `--run-local` 或 `--run-api`（或一起 `--run-both`）。

对比维度：
  - VLM 是否输出合法 JSON
  - Schema 是否通过
  - 规划的下游模型列表
  - workflow_steps 数量
  - step instruction 参数差异
  - scene_understanding 文本相似度
  - VLM 调用耗时（per-iteration）
  - 最终检测数量
  - 验证时剔除的假阳性数量
  - 整体 ok 状态

用法：
  # 仅做规划分析（不真实调用 VLM）
  python scripts/compare_vlm_backends.py \\
      --run-dir runs/<sample_id> \\
      --out-dir out/ab

  # 真实跑两边（注意：两边都会消耗资源；API 端会消耗额度）
  python scripts/compare_vlm_backends.py \\
      --run-dir runs/<sample_id> \\
      --out-dir out/ab \\
      --run-both \\
      --no-verify --max-iter 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv_silent() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        env_path = ROOT / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
    except ImportError:
        pass


_load_dotenv_silent()


# ---------------------------------------------------------------------------
# 后端适配器：负责构造对应后端的 VLMClient
# ---------------------------------------------------------------------------

def _build_local_vlm():
    from pipeline.vlm import VLMClient
    return VLMClient(backend="local")


def _build_api_vlm():
    if not (os.environ.get("VLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")):
        raise RuntimeError(
            "API 后端需要 VLM_API_KEY 或 DASHSCOPE_API_KEY（已在 .gitignore 里）"
        )
    from pipeline.vlm import VLMClient
    return VLMClient(backend="api")


_BACKENDS = {
    "local": (_build_local_vlm, "需要 QWEN_VL_PATH 和本地 GPU"),
    "api":   (_build_api_vlm,   "需要 VLM_API_KEY 或 DASHSCOPE_API_KEY"),
}


# ---------------------------------------------------------------------------
# 对比
# ---------------------------------------------------------------------------

@dataclass
class BackendResult:
    name: str
    ran: bool
    error: str | None = None
    report: dict = field(default_factory=dict)
    elapsed_vlm_s: float = 0.0
    elapsed_total_s: float = 0.0
    n_vlm_calls: int = 0


def _run_one(backend_name: str, run_dir: str, out_dir: Path,
             no_verify: bool, max_iter: int) -> BackendResult:
    from pipeline import Pipeline
    builder, prereq = _BACKENDS[backend_name]
    print(f"\n[A/B] === Running backend={backend_name} ===")
    print(f"       ({prereq})")
    sub = out_dir / f"out_{backend_name}"
    sub.mkdir(parents=True, exist_ok=True)
    try:
        vlm = builder()
    except Exception as e:
        return BackendResult(name=backend_name, ran=False, error=f"init: {e}")
    started = time.perf_counter()
    try:
        p = Pipeline(vlm=vlm, verbose=False)
        report = p.run_from_adapter(
            run_dir=run_dir,
            out_dir=str(sub),
            verify=not no_verify,
            max_iterations=max_iter,
        )
    except Exception as e:
        return BackendResult(
            name=backend_name,
            ran=False,
            error=f"pipeline: {type(e).__name__}: {e}",
            elapsed_vlm_s=float(getattr(vlm, "elapsed_total_s", 0.0)),
            elapsed_total_s=time.perf_counter() - started,
            n_vlm_calls=int(getattr(vlm, "call_count", 0)),
        )
    finally:
        close = getattr(vlm, "close", None)
        if callable(close):
            close()
    return BackendResult(
        name=backend_name, ran=True, report=report,
        n_vlm_calls=int(getattr(vlm, "call_count", 0)),
        elapsed_vlm_s=float(getattr(vlm, "elapsed_total_s", 0.0)),
        elapsed_total_s=time.perf_counter() - started,
    )


def _plan_signature(plan: dict) -> dict:
    """从 plan 里抽可比对的关键字段。"""
    steps = plan.get("workflow_steps") or []
    sig = {
        "scene_understanding": (plan.get("scene_understanding") or "").strip(),
        "max_iterations": plan.get("max_iterations"),
        "n_steps": len(steps),
        "models": sorted({s.get("model", "?") for s in steps}),
        "step_models": [s.get("model") for s in steps],
        "step_instructions": [
            {k: v for k, v in (s.get("instruction") or {}).items()
             if k not in {"seed_points"}}   # seed_points 太长，省略
            for s in steps
        ],
    }
    return sig


def _compare(
    local: BackendResult | None,
    api: BackendResult | None,
) -> dict:
    local = local or BackendResult(name="local", ran=False, error="not requested")
    api = api or BackendResult(name="api", ran=False, error="not requested")
    diff: dict[str, Any] = {
        "both_ran": local.ran and api.ran,
        "ran": {"local": local.ran, "api": api.ran},
        "ok": {
            "local": (local.report.get("ok") if local.ran else None),
            "api":   (api.report.get("ok") if api.ran else None),
        },
        "errors": {
            "local": local.error,
            "api":   api.error,
        },
        "vlm_calls": {
            "local": local.n_vlm_calls,
            "api":   api.n_vlm_calls,
        },
        "elapsed_vlm_s": {
            "local": round(local.elapsed_vlm_s, 3) if local.ran else None,
            "api": round(api.elapsed_vlm_s, 3) if api.ran else None,
        },
        "elapsed_total_s": {
            "local": round(local.elapsed_total_s, 3) if local.ran else None,
            "api": round(api.elapsed_total_s, 3) if api.ran else None,
        },
        "verifications": {
            "local": len(local.report.get("verifications") or []),
            "api":   len(api.report.get("verifications") or []),
        },
        "n_detections": {
            "local": (local.report.get("downstream") or {}).get("n_detections")
                     if local.ran else None,
            "api":   (api.report.get("downstream") or {}).get("n_detections")
                     if api.ran else None,
        },
    }
    if local.ran and api.ran:
        ls, aps = _plan_signature(local.report.get("vlm_plan") or {}), \
                  _plan_signature(api.report.get("vlm_plan") or {})
        diff["plan"] = {
            "local": ls, "api": aps,
            "models_match": set(ls["models"]) == set(aps["models"]),
            "n_steps_match": ls["n_steps"] == aps["n_steps"],
            "scene_match": ls["scene_understanding"] == aps["scene_understanding"],
            "scene_len_local": len(ls["scene_understanding"]),
            "scene_len_api":   len(aps["scene_understanding"]),
        }
        # 验证阶段假阳剔除
        def _drop_count(rep):
            total = 0
            for v in rep.get("verifications") or []:
                total += len(v.get("filtered_ids") or [])
            return total
        diff["false_positives_dropped"] = {
            "local": _drop_count(local.report),
            "api":   _drop_count(api.report),
        }
    return diff


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A/B 对比 VLM 本地/API 后端",
    )
    p.add_argument("--run-dir", required=True,
                   help="geo_adapter 产出的 runs/<sample_id>/")
    p.add_argument("--out-dir", default="out/ab",
                   help="输出目录；两边各放 out_<backend>/")
    p.add_argument("--run-local", action="store_true",
                   help="真实跑本地后端（需要 QWEN_VL_PATH + GPU）")
    p.add_argument("--run-api", action="store_true",
                   help="真实跑 API 后端（需要 VLM_API_KEY，会产生费用）")
    p.add_argument("--run-both", action="store_true",
                   help="等价于 --run-local --run-api")
    p.add_argument("--no-verify", action="store_true",
                   help="关闭 VLM 验证回环（推荐 A/B 第一轮用）")
    p.add_argument("--max-iter", type=int, default=1,
                   help="验证回环最大迭代数（默认 1）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.run_both:
        args.run_local = args.run_api = True

    if (args.run_local or args.run_api) and not Path(args.run_dir).is_dir():
        print(f"ERROR: --run-dir {args.run_dir} 不存在", file=sys.stderr)
        return 2

    results: dict[str, BackendResult] = {}
    if args.run_local:
        results["local"] = _run_one("local", args.run_dir, out_dir,
                                    args.no_verify, args.max_iter)
    if args.run_api:
        results["api"] = _run_one("api", args.run_dir, out_dir,
                                  args.no_verify, args.max_iter)

    if not results:
        print("[A/B] 没指定 --run-local/--run-api/--run-both，仅做参数规划分析。")
        print("       （说明：规划差异需要真实 VLM 跑一次才能拿到——请加 --run-both）")
        # 只检查下游模型清单差异（不需要 VLM）
        try:
            from pipeline import downstream as ds
            print(f"[A/B] 下游模型清单（与 VLM 无关）: {ds.available_names()}")
        except Exception as e:
            print(f"[A/B] 下游模型清单读取失败: {e}")
        # 写一个空的 comparison.json
        (out_dir / "comparison.json").write_text(
            json.dumps({"ran": False,
                        "hint": "rerun with --run-both to actually call VLM"},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0

    diff = _compare(results.get("local"), results.get("api"))
    out = {
        "ran_backends": list(results.keys()),
        "comparison": diff,
        "results": {
            k: {
                "ran": v.ran, "error": v.error, "report": v.report,
                "n_vlm_calls": v.n_vlm_calls,
                "elapsed_vlm_s": round(v.elapsed_vlm_s, 3),
                "elapsed_total_s": round(v.elapsed_total_s, 3),
            } for k, v in results.items()
        },
    }
    out_p = out_dir / "comparison.json"
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                     encoding="utf-8")

    # 控制台简短摘要
    print(f"\n[A/B] comparison saved: {out_p}")
    for k, v in results.items():
        if v.ran:
            r = v.report
            print(f"  - {k}: ok={r.get('ok')} "
                  f"n_detections={(r.get('downstream') or {}).get('n_detections')} "
                  f"iterations={len(r.get('verifications') or [])} "
                  f"vlm={v.elapsed_vlm_s:.1f}s total={v.elapsed_total_s:.1f}s")
        else:
            print(f"  - {k}: NOT RAN ({v.error})")
    if diff.get("plan"):
        dp = diff["plan"]
        print(f"  plan: models_match={dp['models_match']} "
              f"n_steps_match={dp['n_steps_match']} "
              f"scene_match={dp['scene_match']}")
    return 0 if all(v.ran and v.report.get("ok") for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
