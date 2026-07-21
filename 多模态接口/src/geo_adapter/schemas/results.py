from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class InspectionResult(BaseModel):
    success: bool
    inputs: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PrepareResult(BaseModel):
    success: bool
    output_directory: Path | None = None
    manifest_path: Path | None = None
    request_path: Path | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    run_mode: str | None = None
    horizontal_alignment: str | None = None
    vertical_alignment: str | None = None
    fusion_permission: str | None = None


class ValidationResult(BaseModel):
    success: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    checked_files: int = 0

