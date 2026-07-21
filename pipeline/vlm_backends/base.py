"""VLM 后端抽象基类。

所有后端（local Qwen、OpenAI 兼容 API、未来的 Anthropic 等）都必须实现
`call()` 接口并返回 (raw_text, elapsed_seconds) 二元组，与原 VLMClient
的底层调用完全等价。

`call_json()` / JSON 提取 / Schema 校验 / 一次修复重试都集中在 `pipeline.vlm`
门面上实现，不在 backend 内重复，避免本地和 API 两套逻辑漂移。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class VLMBackend(ABC):
    """所有 VLM 后端必须实现的接口。"""

    #: 后端人类可读名（日志/报告里展示）。
    name: str = "base"

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        images: list,
        user_text: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, float]:
        """底层调用 VLM，返回 (raw_text, elapsed_seconds)。

        参数:
            system_prompt: 系统提示词。
            images:        PIL Image / Path / str / bytes 的列表。
                           空列表表示纯文本请求。
            user_text:     用户消息文本。
            max_new_tokens: 最大生成 token 数。
            temperature:   采样温度；<=0 表示 greedy。

        异常:
            任何不可恢复错误都应该 raise（缺 key、网络挂、模型拒绝、空内容等）。
            错误信息中允许包含：模型名、base URL 主机、图片数、HTTP 状态码、
            服务端错误类型。**不得**包含完整 API key 或 base64 图片。
        """

    # 某些后端需要显式释放资源（GPU 显存、连接池等）。
    def close(self) -> None:  # pragma: no cover - 默认 no-op
        return None

    # 让 backend 支持上下文管理器。
    def __enter__(self) -> "VLMBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
