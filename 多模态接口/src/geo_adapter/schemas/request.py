from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ContentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image", "json"]
    name: str | None = None
    path: str | None = None
    analysis_path: str | None = None
    text_path: str | None = None
    physical_view: str | None = None
    native_shape: list[int] | None = None
    axis_labels: list[str] | None = None
    source_indices: dict[str, Any] | None = None


class Message(BaseModel):
    role: Literal["system", "user"]
    content: list[ContentItem]


class ModelRequest(BaseModel):
    schema_version: str = "1.0"
    model_family: str = "qwen3_vl"
    sample_id: str
    messages: list[Message]
    expected_output_schema: str
