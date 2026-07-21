"""VLM client: 加载 Qwen3-VL 一次，支持 schema-validated JSON 输出+一次自动重试。"""
from __future__ import annotations

import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any

from ._logging import get_logger

_logger = get_logger("vlm")

DEFAULT_MODEL_PATH = os.environ.get("QWEN_VL_PATH")


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


@dataclass
class VLMResponse:
    text: str
    data: dict | None
    elapsed_s: float
    attempts: int
    schema_valid: bool
    schema_errors: list[str]


class VLMClient:
    """Qwen3-VL 客户端。惰性加载，进程内单例可复用。"""

    def __init__(self, model_path: str | None = None,
                 dtype: str = "bfloat16", device_map: str = "auto"):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.dtype = dtype
        self.device_map = device_map
        self._model = None
        self._processor = None

    def load(self):
        if self._model is not None:
            return
        if not self.model_path:
            raise RuntimeError(
                "Qwen3-VL 模型路径未配置：请设置环境变量 QWEN_VL_PATH 指向 "
                "Qwen3-VL-8B-Instruct 目录，或在 VLMClient(model_path=...) 显式传入。"
            )
        warnings.filterwarnings("ignore")
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch_dtype = getattr(torch, self.dtype)
        _logger.info(f"[VLM] loading {self.model_path} ...")
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch_dtype,
            device_map=self.device_map, trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True,
        )

    def call(self, system_prompt: str, images: list, user_text: str,
             max_new_tokens: int = 4096, temperature: float = 0.0) -> tuple[str, float]:
        """底层调用：返回 (raw_text, elapsed_s)。"""
        import torch
        self.load()
        content: list[dict] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if images:
            inputs = self._processor(text=text, images=images,
                                     return_tensors="pt").to(self._model.device)
        else:
            inputs = self._processor.tokenizer(text, return_tensors="pt").to(
                self._model.device)
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens, repetition_penalty=1.1,
        )
        if temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=0.95))
        t0 = time.time()
        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)
        resp = self._processor.decode(output[0], skip_special_tokens=True)
        if "assistant" in resp:
            resp = resp.split("assistant")[-1].strip()
        return resp, time.time() - t0

    def call_json(self, system_prompt: str, images: list, user_text: str,
                  schema: dict | None = None, max_new_tokens: int = 4096,
                  temperature: float = 0.0) -> VLMResponse:
        """调用 VLM 期望 JSON 输出。带 schema 时校验失败会用错误信息作反馈重试一次。

        retry 时把上一份 JSON 全文一起发回去，让 VLM 安全地局部修正（而不是重新生成一份空壳）。
        """
        from schemas.output_schemas import validate_output
        raw, elapsed = self.call(system_prompt, images, user_text,
                                 max_new_tokens=max_new_tokens,
                                 temperature=temperature)
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
        retry_text = (f"{user_text}\n\n"
                      f"上一次输出如下，只被 schema 拒绝了一处:\n{prev_json}\n\n"
                      f"仅有这一处需要修正: {err_msg}\n"
                      f"请保持其他字段不变，仅修正报错的字段后输出完整的 JSON。"
                      f"不要留空字符串，不要输出空壳。")
        raw2, e2 = self.call(system_prompt, images, retry_text,
                             max_new_tokens=max_new_tokens,
                             temperature=0.0)   # retry 用 decisive
        data2 = extract_json(raw2)
        if data2 is None:
            return VLMResponse(raw2, None, elapsed + e2, 2, False,
                               errs + ["retry: JSON extraction failed"])
        ok2, errs2 = validate_output(schema, data2)
        # retry 也 schema 不过时，仍保留 data（上层可宽松继续）
        return VLMResponse(raw2, data2, elapsed + e2, 2, ok2,
                           [] if ok2 else errs2)
