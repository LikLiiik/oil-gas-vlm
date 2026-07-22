"""OpenAI 兼容多模态 API 后端。

支持 DashScope（compatible-mode 端点）、OpenAI、vLLM、Together、Anthropic-compatible
代理等所有走 `client.chat.completions.create(...)` 的服务。

设计要点：
- **不** import torch / transformers / accelerate——API 模式可在纯 CPU / 无 GPU 环境运行。
- openai SDK 在 `_build_client()` 内 import；模块被 import 时不强制依赖。
- 图片统一编码为 base64 data URL（PNG 或 JPEG），原图传入路径由 `VLM_API_MAX_IMAGE_EDGE`
  控制；`0` 表示不主动缩放（推荐第一轮 A/B 用原图）。
- 网络错误用指数退避有限重试（默认 2 次，0 表示不重试），与 `call_json()`
  层面的 schema 重试**互不耦合**——一个网络层、一个 schema 层，分别独立计数。
- 错误信息包含：模型名、base URL 主机、图片数、HTTP 状态、服务端错误类型。
  **绝不**打印 API key 或完整 base64 图片。
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .base import VLMBackend

_logger = logging.getLogger("pipeline.vlm.api")


# 哪些 HTTP 状态属于"可重试"。401 / 4xx 大多是配置错误，重试没意义。
_RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504}


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(
            f"env {name}={raw!r} is not a valid integer (default={default})"
        ) from e


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(
            f"env {name}={raw!r} is not a valid float (default={default})"
        ) from e


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# ---------------------------------------------------------------------------
# 图片编码
# ---------------------------------------------------------------------------

def _is_pil_image(obj: Any) -> bool:
    """检测 PIL.Image.Image，但不强制 import PIL（避免无谓依赖）。"""
    cls = obj.__class__
    return cls.__module__.startswith("PIL.") and cls.__name__ == "Image"


def _resize_keep_aspect(img, max_edge: int):
    """如果 max_edge>0 且图超过限制，按长边等比缩放。"""
    if max_edge is None or max_edge <= 0:
        return img
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return img
    scale = max_edge / float(long_edge)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    # Pillow 兼容性：Pillow 9+ 有 Image.Resampling；旧版有 Image.LANCZOS
    resample = getattr(img, "Resampling", None)
    if resample is None:
        from PIL import Image as _PILImage
        resample = getattr(_PILImage, "Resampling", None) or _PILImage.LANCZOS
    return img.resize(new_size, resample.LANCZOS)


def _encode_pil_to_data_url(
    img,
    image_format: str = "PNG",
    jpeg_quality: int = 95,
    max_edge: int = 0,
) -> tuple[str, tuple[int, int], int]:
    """把 PIL.Image 编码为 data URL，返回 (data_url, (w, h), encoded_bytes)。

    - 默认 PNG；可选 JPEG。WebP 也支持。
    - `max_edge>0` 时先等比缩放。
    """
    fmt = (image_format or "PNG").upper()
    if fmt not in {"PNG", "JPEG", "JPG", "WEBP"}:
        raise ValueError(
            f"VLM_API_IMAGE_FORMAT must be one of PNG|JPEG|WEBP, got {image_format!r}"
        )
    if fmt == "JPG":
        fmt = "JPEG"
    img = _resize_keep_aspect(img, max_edge)
    # PNG / WebP 需要 RGBA 或 RGB；JPEG 不支持 alpha
    if fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    elif fmt == "PNG" and img.mode not in ("RGB", "RGBA", "L", "P"):
        # 其他模式（如 CMYK）转 RGB
        img = img.convert("RGB")

    buf = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    if fmt == "JPEG":
        save_kwargs["quality"] = int(jpeg_quality)
    img.save(buf, format=fmt, **save_kwargs)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    mime = "image/jpeg" if fmt == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}", img.size, len(raw)


def _encode_path_to_data_url(
    path: str | Path,
    image_format: str = "PNG",
    jpeg_quality: int = 95,
    max_edge: int = 0,
) -> tuple[str, tuple[int, int], int]:
    """从磁盘读图编码为 data URL。"""
    from PIL import Image  # 局部 import

    p = Path(path)
    with Image.open(p) as img:
        # 读后再 reopen 一份（with 关闭后无法 save）
        img.load()
        return _encode_pil_to_data_url(
            img,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            max_edge=max_edge,
        )


def _normalize_image(
    img: Any,
    image_format: str,
    jpeg_quality: int,
    max_edge: int,
    idx: int,
) -> tuple[str, tuple[int, int], int]:
    """把单个输入归一为 (data_url, (w, h), encoded_bytes)。

    支持:
    - PIL.Image.Image
    - pathlib.Path / 字符串路径
    - bytes（按字节内容直接当作文件读；要求 PIL 能解）
    """
    if _is_pil_image(img):
        return _encode_pil_to_data_url(
            img, image_format=image_format,
            jpeg_quality=jpeg_quality, max_edge=max_edge,
        )
    if isinstance(img, (str, Path)):
        return _encode_path_to_data_url(
            img,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            max_edge=max_edge,
        )
    if isinstance(img, (bytes, bytearray, memoryview)):
        from PIL import Image  # 局部 import
        bio = io.BytesIO(bytes(img))
        with Image.open(bio) as im:
            im.load()
            return _encode_pil_to_data_url(
                im, image_format=image_format,
                jpeg_quality=jpeg_quality, max_edge=max_edge,
            )
    raise TypeError(
        f"unsupported image type at index {idx}: "
        f"{type(img).__module__}.{type(img).__name__}"
    )


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _resolve_api_key() -> str:
    """密钥优先级：VLM_API_KEY > DASHSCOPE_API_KEY。"""
    for name in ("VLM_API_KEY", "DASHSCOPE_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v
    raise RuntimeError(
        "API 后端缺少密钥。请设置环境变量 VLM_API_KEY 或 DASHSCOPE_API_KEY。"
    )


def _safe_host(base_url: str) -> str:
    """提取 base URL 的 host 部分用于日志，避免泄漏路径/query。"""
    try:
        return urlparse(base_url).hostname or base_url
    except Exception:
        return base_url


# ---------------------------------------------------------------------------
# 后端实现
# ---------------------------------------------------------------------------

class OpenAICompatibleVLMBackend(VLMBackend):
    """OpenAI 兼容多模态后端。"""

    name = "api"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
        json_mode: bool | None = None,
        image_format: str | None = None,
        jpeg_quality: int | None = None,
        max_image_edge: int | None = None,
    ):
        # 基础配置（每个参数都允许从 env 兜底）
        self.api_key = api_key or _resolve_api_key()
        self.base_url = (
            base_url
            or _env("VLM_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model or _env("VLM_MODEL", "qwen3-vl-plus")
        self.timeout = float(timeout if timeout is not None
                             else _env_float("VLM_TIMEOUT", 180.0))
        self.max_tokens = int(max_tokens if max_tokens is not None
                              else _env_int("VLM_MAX_TOKENS", 6144))
        self.temperature = float(temperature if temperature is not None
                                 else _env_float("VLM_TEMPERATURE", 0.1))
        self.max_retries = int(max_retries if max_retries is not None
                               else _env_int("VLM_API_MAX_RETRIES", 2))
        self.json_mode = bool(json_mode if json_mode is not None
                              else _env_bool("VLM_API_JSON_MODE", False))
        self.image_format = (image_format
                             or _env("VLM_API_IMAGE_FORMAT", "PNG")).upper()
        self.jpeg_quality = int(jpeg_quality if jpeg_quality is not None
                                else _env_int("VLM_API_JPEG_QUALITY", 95))
        self.max_image_edge = int(max_image_edge if max_image_edge is not None
                                  else _env_int("VLM_API_MAX_IMAGE_EDGE", 0))
        if self.image_format not in {"PNG", "JPEG", "JPG", "WEBP"}:
            raise ValueError(
                f"VLM_API_IMAGE_FORMAT must be PNG|JPEG|WEBP, got {self.image_format!r}"
            )
        if self.max_retries < 0:
            raise ValueError("VLM_API_MAX_RETRIES must be >= 0")
        if self.timeout <= 0:
            raise ValueError("VLM_TIMEOUT must be > 0")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("VLM_API_JPEG_QUALITY must be between 1 and 100")
        if self.max_image_edge < 0:
            raise ValueError("VLM_API_MAX_IMAGE_EDGE must be >= 0")

        self._client = None  # 惰性建连

    # ---- OpenAI client 懒加载 -------------------------------------------

    def _build_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "API 后端需要 `openai` 包。请运行：pip install openai>=1.40.0"
            ) from e
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            # 重试只由本类下面的显式退避循环控制，避免 SDK 默认重试叠加。
            max_retries=0,
        )
        return self._client

    def close(self) -> None:
        """关闭 OpenAI SDK 的底层 HTTP 连接池。"""
        client, self._client = self._client, None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    # ---- 图片预处理 ------------------------------------------------------

    def _encode_images(self, images: list) -> list[dict]:
        """按 images 顺序编码为 OpenAI 多模态 content 列表。"""
        if images is None:
            images = []
        if not isinstance(images, list):
            raise TypeError(
                f"`images` must be a list, got {type(images).__name__}"
            )
        if len(images) == 0:
            return []
        out: list[dict] = []
        for i, im in enumerate(images):
            try:
                data_url, size, n_bytes = _normalize_image(
                    im,
                    image_format=self.image_format,
                    jpeg_quality=self.jpeg_quality,
                    max_edge=self.max_image_edge,
                    idx=i,
                )
            except Exception as e:
                raise RuntimeError(
                    f"failed to encode image #{i} (type="
                    f"{type(im).__module__}.{type(im).__name__}): {e}"
                ) from e
            _logger.info(
                "[VLM-API] image #%d: orig/after_size=%s, encoded=%d bytes, "
                "format=%s",
                i, size, n_bytes, self.image_format,
            )
            out.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })
        return out

    # ---- 核心调用 --------------------------------------------------------

    def call(
        self,
        system_prompt: str,
        images: list,
        user_text: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, float]:
        if not self.api_key:
            raise RuntimeError("API key missing (VLM_API_KEY / DASHSCOPE_API_KEY)")
        if not self.base_url:
            raise RuntimeError("VLM_BASE_URL is empty")

        image_contents = self._encode_images(images or [])
        user_content: list[dict] = list(image_contents)
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 客户端允许服务端忽略我们不识别的参数。`response_format` 单独处理。
        create_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            max_tokens=int(max_new_tokens or self.max_tokens),
        )
        # temperature：API 端通常要求 float；<=0 用 0 表示 greedy
        try:
            t = float(temperature) if temperature is not None else self.temperature
        except (TypeError, ValueError):
            t = self.temperature
        create_kwargs["temperature"] = max(0.0, t)

        if self.json_mode:
            create_kwargs["response_format"] = {"type": "json_object"}

        client = self._build_client()
        host = _safe_host(self.base_url)
        n_imgs = len(image_contents)

        last_err: Exception | None = None
        # 初次 + 最多 max_retries 次
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = client.chat.completions.create(**create_kwargs)
                elapsed = time.time() - t0
                text = self._extract_text(resp)
                if not text:
                    # 仍然消耗一次；让上层（call_json）走 schema retry
                    raise RuntimeError(
                        f"[VLM-API] empty content from model={self.model} "
                        f"(host={host}, images={n_imgs}, elapsed={elapsed:.1f}s)"
                    )
                return text, elapsed
            except Exception as e:
                elapsed = time.time() - t0
                retryable, status, err_type = self._classify_error(e)
                last_err = e
                _logger.warning(
                    "[VLM-API] call failed: model=%s host=%s images=%d "
                    "status=%s type=%s attempt=%d/%d elapsed=%.1fs err=%s",
                    self.model, host, n_imgs, status, err_type,
                    attempt + 1, self.max_retries + 1, elapsed, e,
                )
                if not retryable or attempt >= self.max_retries:
                    break
                # 指数退避：1s, 2s, 4s ... 上限 8s
                backoff = min(8.0, 1.0 * (2 ** attempt))
                time.sleep(backoff)
                continue
        # 用尽了：抛最后一次的错（带可读包装）
        assert last_err is not None
        status, err_type = self._classify_error(last_err)[1:]
        raise RuntimeError(
            f"[VLM-API] call failed after retries: model={self.model} "
            f"host={host} images={n_imgs} status={status} type={err_type} "
            f"err={last_err}"
        ) from last_err

    # ---- 响应解析与错误分类 ----------------------------------------------

    @staticmethod
    def _extract_text(resp: Any) -> str:
        """从 OpenAI ChatCompletion 响应中抽出第一个 choice 的文本。"""
        try:
            choices = getattr(resp, "choices", None) or []
        except Exception:
            choices = []
        if not choices:
            return ""
        first = choices[0]
        msg = getattr(first, "message", None)
        if msg is None:
            # 字典风格
            if isinstance(first, dict):
                msg = first.get("message") or {}
                content = msg.get("content")
            else:
                return ""
        else:
            content = getattr(msg, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # content 可能是 list[dict]（多模态部分）— 只取 text 段拼接
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                    if t:
                        parts.append(t)
                else:
                    t = getattr(part, "text", None)
                    if t:
                        parts.append(t)
            return "\n".join(parts)
        # 其它类型：兜底强转
        return str(content)

    @staticmethod
    def _classify_error(e: Exception) -> tuple[bool, int | None, str]:
        """分类错误：返回 (retryable, http_status, err_type_name)。"""
        name = type(e).__name__
        # openai SDK 1.x 的异常：APIStatusError(401/403/404/429/5xx)、
        # APITimeoutError、APIConnectionError、BadRequestError 等
        status: int | None = None
        for attr in ("status_code", "http_status", "code"):
            v = getattr(e, attr, None)
            if isinstance(v, int):
                status = v
                break
        if status is None:
            # 从 message 里抽 "Error code: 429" 这种
            msg = str(e) or ""
            for tok in ("Error code: ", "HTTP/1.1 "):
                if tok in msg:
                    try:
                        after = msg.split(tok, 1)[1].split(" ", 1)[0]
                        status = int("".join(ch for ch in after if ch.isdigit())[:3] or 0) or None
                    except Exception:
                        pass
        retryable = False
        lname = name.lower()
        if "timeout" in lname:
            retryable = True
        elif "connection" in lname:
            retryable = True
        elif status is not None and status in _RETRYABLE_HTTP:
            retryable = True
        return retryable, status, name
