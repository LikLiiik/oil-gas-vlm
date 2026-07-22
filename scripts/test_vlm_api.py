"""VLM API 后端真实冒烟测试。

⚠️ 本脚本会调用真实 API 并**可能产生费用**。只在设置了 VLM_API_KEY /
DASHSCOPE_API_KEY 时才会运行。

用法示例：

    # PowerShell
    $env:VLM_BACKEND = "api"
    $env:VLM_API_KEY  = "<填写在本机，不要提交>"
    $env:VLM_BASE_URL = "https://...compatible-mode/v1"
    $env:VLM_MODEL    = "qwen3-vl-plus"
    python scripts/test_vlm_api.py --run-dir runs/<sample_id> --max-images 2

    # bash
    VLM_BACKEND=api VLM_API_KEY=... python scripts/test_vlm_api.py \\
        --run-dir runs/<sample_id> --max-images 2

不带 --run-dir 时，使用一张临时合成的 256×256 测试图，验证最基本的多模态通路。
带 --run-dir 时，从 geo_adapter 生成的 run 目录里读前 N 张图。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 让 `python scripts/test_vlm_api.py` 直接运行也能找到 pipeline 包
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv_silent() -> None:
    """自动从 .env 加载（如果存在）。失败也无所谓——env 里已有值优先。"""
    try:
        from dotenv import load_dotenv  # type: ignore
        env_path = ROOT / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
    except ImportError:
        pass  # 没装 dotenv 也不报错


_load_dotenv_silent()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VLM API 真实冒烟测试（会产生费用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", type=str, default=None,
                   help="geo_adapter 产出的 runs/<sample_id> 目录；"
                        "省略则用一张临时合成图")
    p.add_argument("--max-images", type=int, default=2,
                   help="最多发给 VLM 的图片数（默认 2）")
    p.add_argument("--out-dir", type=str, default="out/api_smoke",
                   help="保存原始响应和解析结果的目录")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--model", type=str, default=None,
                   help="覆盖 VLM_MODEL（默认从 env 读）")
    return p


def _warn_cost(model: str, n_images: int) -> None:
    print()
    print("=" * 60)
    print("  WARNING: 本次调用会产生 API 费用")
    print(f"  model={model}  images={n_images}")
    print("  中断组合键 Ctrl+C")
    print("=" * 60)
    print()


def _collect_images(run_dir: str | None, max_images: int) -> list:
    """从 run_dir 收集最多 max_images 张 PIL Image；没有就合成 1 张。"""
    import numpy as np
    from PIL import Image

    if run_dir:
        run_dir = Path(run_dir)
        manifest_p = run_dir / "manifest.json"
        if not manifest_p.is_file():
            raise FileNotFoundError(f"manifest.json not found in {run_dir}")
        # 简单扫 assets/*/*.png（按 mtime 排序）
        pngs = sorted(run_dir.glob("assets/*/*.png"))
        pngs = [p for p in pngs if p.is_file()]
        if not pngs:
            raise FileNotFoundError(f"no .png under {run_dir}/assets/")
        pngs = pngs[:max_images]
        return [(Image.open(p).convert("RGB"), p) for p in pngs]

    # 合成一张图：模拟地震剖面风格（黑底+亮线）
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[60, :, 0] = arr[60, :, 1] = arr[60, :, 2] = 200
    arr[120, :, :] = 150
    arr[180, :, 0] = 100
    img = Image.fromarray(arr)
    return [(img, Path("(synthetic)"))]


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    # 安全闸：没 key 就不跑
    if not (os.environ.get("VLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")):
        print("ERROR: 缺少 API key。请先设置 VLM_API_KEY 或 DASHSCOPE_API_KEY。",
              file=sys.stderr)
        return 2
    if (os.environ.get("VLM_BACKEND") or "local") != "api":
        # 不强制覆盖：让用户自己显式开
        print("WARN: VLM_BACKEND != 'api'，脚本会临时把它设为 'api' 再跑。",
              file=sys.stderr)
        os.environ["VLM_BACKEND"] = "api"

    if args.model:
        os.environ["VLM_MODEL"] = args.model

    from pipeline.vlm import VLMClient, VLMResponse  # noqa: E402

    images = _collect_images(args.run_dir, args.max_images)
    n = len(images)
    model = os.environ.get("VLM_MODEL", "qwen3-vl-plus")
    _warn_cost(model, n)

    vlm = VLMClient(backend="api")
    print(f"[smoke] backend={vlm.backend_name} model={vlm.backend.model} "
          f"base_url_host={vlm.backend.base_url.split('//', 1)[-1].split('/', 1)[0]}")

    pil_list = [im for im, _p in images]
    paths = [str(p) for _im, p in images]

    system = ("你是一个乐于助人的助手。回答要简洁、准确。"
              "如果用户让你输出 JSON，请只输出 JSON，不要额外解释。")
    user_text = (
        f"请用 1-2 句话描述这 {n} 张图（地质/地震/测井场景）。"
        "然后输出一个 JSON 块，结构: {\"ok\": true, \"n_images\": <int>, "
        "\"keywords\": [<3-5 个关键词>]}。JSON 放在最后。"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 底层 call()
    text, elapsed = vlm.call(
        system, pil_list, user_text,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    raw_p = out_dir / "raw_response.txt"
    raw_p.write_text(text or "", encoding="utf-8")
    print(f"[smoke] raw call: elapsed={elapsed:.1f}s, "
          f"text_len={len(text or '')}, saved={raw_p}")

    # 2) call_json() 走完整 schema retry 路径
    schema = {
        "type": "object",
        "required": ["ok", "n_images", "keywords"],
        "properties": {
            "ok": {"type": "boolean"},
            "n_images": {"type": "integer"},
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
    }
    resp: VLMResponse = vlm.call_json(
        system, pil_list, user_text,
        schema=schema,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    summary = {
        "ok": resp.data is not None and resp.schema_valid,
        "schema_valid": resp.schema_valid,
        "attempts": resp.attempts,
        "elapsed_s": round(resp.elapsed_s, 2),
        "n_images_sent": n,
        "image_paths": paths,
        "model": vlm.backend.model,
        "base_url_host": vlm.backend.base_url.split("//", 1)[-1].split("/", 1)[0],
        "schema_errors": resp.schema_errors,
        "data": resp.data,
    }
    out_p = out_dir / "summary.json"
    out_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"[smoke] call_json: schema_valid={resp.schema_valid} "
          f"attempts={resp.attempts} elapsed={resp.elapsed_s:.1f}s")
    print(f"[smoke] saved summary: {out_p}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
