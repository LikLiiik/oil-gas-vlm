"""VLM 门面：本地 / API 双后端 + JSON 提取 + Schema 校验 + 一次修复重试。

设计原则：
- 公开 API 与重构前 100% 兼容：
    - ``VLMClient`` 类名、构造签名（model_path / dtype / device_map）保持可用
    - ``call(system_prompt, images, user_text, max_new_tokens, temperature) -> (text, elapsed)``
    - ``call_json(system_prompt, images, user_text, schema, max_new_tokens, temperature) -> VLMResponse``
    - ``VLMResponse`` 字段保持
    - ``extract_json()`` 函数保持
- 后端切换完全内部化——上层 (orchestrator / agents) 不需要任何修改。
- ``call_json()`` 的 JSON 提取、Schema 校验、一次结构化修复重试统一在门面上
  实现，避免本地/API 两套逻辑漂移。
- API 模式下，**不会** import torch / transformers；只有本地后端才惰性加载。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from ._logging import get_logger
from .vlm_backends import (
    LocalQwenVLMBackend,
    OpenAICompatibleVLMBackend,
    VLMBackend,
)

_logger = get_logger("vlm")

DEFAULT_MODEL_PATH = os.environ.get("QWEN_VL_PATH")
DEFAULT_BACKEND = "local"


# ---------------------------------------------------------------------------
# JSON 提取（与原实现完全等价）
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """从 VLM 文本响应里抽出第一个合法的 dict JSON。"""
    for m in re.finditer(r'\{', text):
        start, depth, in_str, esc, end = m.start(), 0, False, False, -1
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == '\\':
                esc = True
                continue
            if c == '"' and not esc:
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            if depth == 0:
                end = i + 1
                break
        if end > start:
            try:
                obj = json.loads(text[start:end])
                if isinstance(obj, dict) and len(obj) >= 1:
                    return obj
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# 响应 dataclass
# ---------------------------------------------------------------------------

@dataclass
class VLMResponse:
    text: str
    data: dict | None
    elapsed_s: float
    attempts: int
    schema_valid: bool
    schema_errors: list[str]


# ---------------------------------------------------------------------------
# 后端选择
# ---------------------------------------------------------------------------

_VALID_BACKENDS = ("local", "api")


def _select_backend(
    backend: str | None,
    model_path: str | None,
    dtype: str,
    device_map: str,
) -> VLMBackend:
    """按优先级解析后端：显式参数 > VLM_BACKEND env > 安全默认 local。

    显式默认 local：避免用户没意识到时产生 API 费用。
    """
    name = (backend or os.environ.get("VLM_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if name not in _VALID_BACKENDS:
        raise ValueError(
            f"unknown VLM backend: {name!r}. Valid: {', '.join(_VALID_BACKENDS)}"
        )
    if name == "local":
        return LocalQwenVLMBackend(
            model_path=model_path or DEFAULT_MODEL_PATH,
            dtype=dtype, device_map=device_map,
        )
    # api
    return OpenAICompatibleVLMBackend()


# ---------------------------------------------------------------------------
# 统一 call_json（含 Schema 重试）
# ---------------------------------------------------------------------------

def _call_json_with_backend(
    backend: VLMBackend,
    system_prompt: str,
    images: list,
    user_text: str,
    schema: dict | None = None,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
) -> VLMResponse:
    """委托给后端的 call()，本地做 JSON 提取 + Schema 校验 + 一次修复重试。"""
    from schemas.output_schemas import validate_output

    raw, elapsed = backend.call(
        system_prompt, images, user_text,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    data = extract_json(raw)
    if data is None:
        return VLMResponse(raw, None, elapsed, 1, False,
                           ["failed to extract JSON from response"])
    if schema is None:
        return VLMResponse(raw, data, elapsed, 1, True, [])
    ok, errs = validate_output(schema, data)
    if ok:
        return VLMResponse(raw, data, elapsed, 1, True, [])
    # 一次结构化重试：带上上一份 JSON 全文 + 精确错误
    prev_json = json.dumps(data, ensure_ascii=False, indent=2)
    err_msg = "; ".join(errs)[:600]
    retry_text = (
        f"{user_text}\n\n"
        f"上一次输出如下，只被 schema 拒绝了一处:\n{prev_json}\n\n"
        f"仅有这一处需要修正: {err_msg}\n"
        f"请保持其他字段不变，仅修正报错的字段后输出完整的 JSON。"
        f"不要留空字符串，不要输出空壳。"
    )
    raw2, e2 = backend.call(
        system_prompt, images, retry_text,
        max_new_tokens=max_new_tokens, temperature=0.0,   # retry 用 decisive
    )
    data2 = extract_json(raw2)
    if data2 is None:
        return VLMResponse(raw2, None, elapsed + e2, 2, False,
                           errs + ["retry: JSON extraction failed"])
    ok2, errs2 = validate_output(schema, data2)
    # retry 也 schema 不过时，仍保留 data（上层可宽松继续）
    return VLMResponse(raw2, data2, elapsed + e2, 2, ok2,
                       [] if ok2 else errs2)


# ---------------------------------------------------------------------------
# 公开门面：VLMClient
# ---------------------------------------------------------------------------

class VLMClient:
    """对外门面：构造时选后端，call/call_json 全部委托给 backend。

    用法与原 VLMClient 一致：

        vlm = VLMClient()                          # 走 env VLM_BACKEND，默认 local
        vlm = VLMClient(backend="api")             # 强制 API
        text, t = vlm.call(sys, [img], user_text)  # 底层
        resp = vlm.call_json(sys, [img], user_text, schema=...)  # 含 schema 重试
    """

    def __init__(
        self,
        model_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        backend: str | None = None,
    ):
        self.backend_name = (
            (backend or os.environ.get("VLM_BACKEND") or DEFAULT_BACKEND).strip().lower()
        )
        if self.backend_name not in _VALID_BACKENDS:
            raise ValueError(
                f"unknown VLM backend: {self.backend_name!r}. "
                f"Valid: {', '.join(_VALID_BACKENDS)}"
            )
        # 兼容旧字段
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.dtype = dtype
        self.device_map = device_map
        # 真正干活的 backend
        self._impl: VLMBackend = _select_backend(
            backend=self.backend_name,
            model_path=model_path,
            dtype=dtype,
            device_map=device_map,
        )

    # ---- 透明代理 -------------------------------------------------------

    @property
    def backend(self) -> VLMBackend:
        return self._impl

    def call(
        self,
        system_prompt: str,
        images: list,
        user_text: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, float]:
        """底层调用，返回 (raw_text, elapsed_s)。"""
        return self._impl.call(
            system_prompt, images, user_text,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

    def call_json(
        self,
        system_prompt: str,
        images: list,
        user_text: str,
        schema: dict | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> VLMResponse:
        """调用 VLM 期望 JSON 输出。带 schema 时校验失败会用错误信息作反馈重试一次。"""
        return _call_json_with_backend(
            self._impl,
            system_prompt=system_prompt, images=images, user_text=user_text,
            schema=schema, max_new_tokens=max_new_tokens, temperature=temperature,
        )

    # ---- 上下文管理 ------------------------------------------------------

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "VLMClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "VLMClient",
    "VLMResponse",
    "extract_json",
    "VLMBackend",
    "LocalQwenVLMBackend",
    "OpenAICompatibleVLMBackend",
]


# 显式让类型检查器 / 静态分析看到默认导出
_ = Any
