"""本地后端：Qwen3-VL 推理（与原 VLMClient.load()/call() 行为一致）。

保持惰性加载策略：
- 类被 import / 实例化时不会触发 torch / transformers
- 只有真正调用 `load()` 之后（首次 `call()` 自动调用）才 import 重模型
- 这样 API 模式可以完全避开本地权重导入

行为兼容性：
- 行为与原 `pipeline/vlm.py` 的 `VLMClient.call()` 完全一致
- 温度 <= 0 → greedy；> 0 → do_sample + top_p=0.95
- decode 后仍做一次 "assistant" 分割，避免把 prompt 一起回吐
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Any

from .base import VLMBackend


class LocalQwenVLMBackend(VLMBackend):
    """本地 Qwen3-VL 后端。"""

    name = "local"

    def __init__(
        self,
        model_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        self.model_path = model_path or os.environ.get("QWEN_VL_PATH")
        self.dtype = dtype
        self.device_map = device_map
        self._model = None
        self._processor = None
        self._loaded = False

    # ---- 加载控制（惰性） ------------------------------------------------

    def load(self) -> None:
        """惰性加载模型权重。多次调用安全。"""
        if self._loaded:
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
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        self._loaded = True

    # ---- 核心调用 --------------------------------------------------------

    def call(
        self,
        system_prompt: str,
        images: list,
        user_text: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, float]:
        """本地推理。返回 (raw_text, elapsed_s)。"""
        import torch  # 局部 import：API 模式完全避开

        self.load()
        content: list[dict] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if images:
            inputs = self._processor(
                text=text, images=images, return_tensors="pt",
            ).to(self._model.device)
        else:
            inputs = self._processor.tokenizer(text, return_tensors="pt").to(
                self._model.device
            )

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
        elapsed = time.time() - t0
        resp = self._processor.decode(output[0], skip_special_tokens=True)
        if "assistant" in resp:
            resp = resp.split("assistant")[-1].strip()
        return resp, elapsed
