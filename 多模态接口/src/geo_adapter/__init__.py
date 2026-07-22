"""GeoMultimodal Input Adapter public API."""

__version__ = "0.1.0"

from .pipeline import inspect_geo_sample, prepare_geo_sample, validate_run
from .schemas.results import InspectionResult, PrepareResult, ValidationResult

__all__ = [
    "InspectionResult",
    "PrepareResult",
    "ValidationResult",
    "inspect_geo_sample",
    "prepare_geo_sample",
    "validate_run",
]

