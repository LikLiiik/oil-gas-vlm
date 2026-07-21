"""下游模型基类 + 注册表。VLM 从注册表里选一个模型调用。"""
from __future__ import annotations

from typing import Any, Protocol


class DownstreamModel(Protocol):
    name: str
    description: str
    required_fields: list[str]
    output_shape: str

    def detect(self, instruction: dict, image: Any = None,
               context: dict | None = None) -> list[dict]: ...


_REGISTRY: dict[str, DownstreamModel] = {}


def register(model: DownstreamModel) -> None:
    """注册下游模型。同名会覆盖，方便替换 mock/真实实现。"""
    _REGISTRY[model.name] = model


def get(name: str) -> DownstreamModel | None:
    return _REGISTRY.get(name)


def available_names() -> list[str]:
    return sorted(_REGISTRY.keys())


def available_models_desc() -> str:
    """生成给 VLM 看的下游模型清单文本。"""
    lines = ["可用的下游模型:"]
    for i, name in enumerate(available_names(), 1):
        m = _REGISTRY[name]
        lines.append(f"{i}. {name}: {m.description}")
        lines.append(f"   必需字段: {m.required_fields}")
        lines.append(f"   输出: {m.output_shape}")
    return "\n".join(lines)
