"""集成测试：跑 LoopAgent（地震+测井），把结果落盘。

生产逻辑在 pipeline/ 里；本文件只负责组装图像 + 调 Pipeline + 落盘。
    CUDA_VISIBLE_DEVICES=1 python test/test_loop.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import Pipeline
from test.data import generate_log, generate_seismic


def main() -> int:
    out_path = os.environ.get("LOOP_OUTPUT", "/tmp/autonomous_agent_results.json")
    p = Pipeline()

    print("=" * 60)
    print("AUTONOMOUS AGENT: Seismic Section")
    print("=" * 60)
    seismic_result = p.run_seismic(generate_seismic(seed=42))

    print("\n\n" + "=" * 60)
    print("AUTONOMOUS AGENT: Well Log")
    print("=" * 60)
    log_result = p.run_log(generate_log(seed=42))

    print("\n\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name, r in [("seismic", seismic_result), ("log", log_result)]:
        if not r or not r.plan:
            print(f"  {name}: FAILED ({r.errors if r else 'no result'})")
            continue
        steps = r.plan.get("workflow_steps", [])
        models = [s.get("model") for s in steps]
        print(f"  {name}:")
        print(f"    Scene: {r.plan.get('scene_understanding','')[:120]}")
        print(f"    Models chosen: {models}")
        print(f"    Steps: {len(steps)} | Results: {len(r.results)} "
              f"| Verifications: {len(r.verifications)}")

    payload = {"seismic": seismic_result.to_dict(),
               "log":     log_result.to_dict()}
    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
