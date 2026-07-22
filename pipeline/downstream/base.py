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


def runtime_status(name: str) -> tuple[bool, str]:
    """Cheap readiness probe that must not download weights or run inference."""
    model = _REGISTRY.get(name)
    if model is None:
        return False, "not registered"
    probe = getattr(model, "runtime_status", None)
    if probe is None:
        return True, "ready"
    try:
        ready, reason = probe()
        return bool(ready), str(reason)
    except Exception as exc:
        return False, f"readiness probe failed: {exc}"


def runnable_names() -> list[str]:
    return [name for name in available_names() if runtime_status(name)[0]]


def available_models_desc() -> str:
    """Generate a runtime-accurate model list without loading large weights."""
    ready_names = runnable_names()
    lines = ["当前运行环境可执行的下游模型:"]
    for i, name in enumerate(ready_names, 1):
        m = _REGISTRY[name]
        lines.append(f"{i}. {name}: {m.description}")
        lines.append(f"   必需字段: {m.required_fields}")
        lines.append(f"   输出: {m.output_shape}")
    unavailable = [
        f"{name} ({runtime_status(name)[1]})"
        for name in available_names()
        if name not in ready_names
    ]
    if unavailable:
        lines.append("不可选择的模型: " + "; ".join(unavailable))
    return "\n".join(lines)
