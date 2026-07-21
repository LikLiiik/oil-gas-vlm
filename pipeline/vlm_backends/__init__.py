"""VLM 后端子包。

对外暴露：
- ``VLMBackend``        抽象基类
- ``LocalQwenVLMBackend`` 本地 Qwen3-VL 推理（惰性加载 transformers/torch）
- ``OpenAICompatibleVLMBackend`` OpenAI 兼容多模态 API（不加载 torch）

后端选择由 ``pipeline.vlm.VLMClient`` 统一处理，使用方一般不需要直接 import 这些类。
"""
from .base import VLMBackend
from .local_qwen import LocalQwenVLMBackend
from .openai_compatible import OpenAICompatibleVLMBackend

__all__ = [
    "VLMBackend",
    "LocalQwenVLMBackend",
    "OpenAICompatibleVLMBackend",
]
