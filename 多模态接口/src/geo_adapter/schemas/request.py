from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ContentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image", "json"]
    name: str | None = None
    path: str | None = None
    text_path: str | None = None
    physical_view: str | None = None


class Message(BaseModel):
    role: Literal["system", "user"]
    content: list[ContentItem]


class ModelRequest(BaseModel):
    schema_version: str = "1.0"
    model_family: str = "qwen3_vl"
    sample_id: str
    messages: list[Message]
    expected_output_schema: str
