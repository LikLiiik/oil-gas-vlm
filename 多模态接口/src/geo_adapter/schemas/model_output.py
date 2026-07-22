from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SeismicObservation(BaseModel):
    type: str
    description: str
    image_name: str
    bbox_xyxy_norm: list[float] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_box(self) -> "SeismicObservation":
        x1, y1, x2, y2 = self.bbox_xyxy_norm
        if not all(0 <= value <= 1 for value in self.bbox_xyxy_norm) or x1 > x2 or y1 > y2:
            raise ValueError("bbox_xyxy_norm 必须在 0..1 且满足 x1<=x2、y1<=y2")
        return self


class SeismicAnalysis(BaseModel):
    available: bool
    observations: list[SeismicObservation] = Field(default_factory=list)


class WellObservation(BaseModel):
    depth_range: list[float] = Field(min_length=2, max_length=2)
    depth_reference: Literal["MD", "TVD", "TVDSS"]
    description: str
    confidence: float = Field(ge=0, le=1)
    evidence_curves: list[str] = Field(default_factory=list)


class WellLogAnalysis(BaseModel):
    available: bool
    observations: list[WellObservation] = Field(default_factory=list)


class CrossModalAnalysis(BaseModel):
    allowed: bool
    alignment_level: str
    conclusion: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)


class RegionOfInterest(BaseModel):
    image_name: str
    bbox_xyxy_norm: list[float] = Field(min_length=4, max_length=4)


class DownstreamPlan(BaseModel):
    task: str = "yolo_world_detection"
    input_images: list[str] = Field(default_factory=list)
    class_prompts: list[str] = Field(default_factory=list)
    regions_of_interest: list[RegionOfInterest] = Field(default_factory=list)
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)


class Uncertainty(BaseModel):
    level: Literal["low", "medium", "high", "unknown"]
    reasons: list[str] = Field(default_factory=list)


class ExpectedModelOutput(BaseModel):
    sample_id: str
    seismic_analysis: SeismicAnalysis
    well_log_analysis: WellLogAnalysis
    cross_modal_analysis: CrossModalAnalysis
    downstream_plan: DownstreamPlan
    uncertainty: Uncertainty

