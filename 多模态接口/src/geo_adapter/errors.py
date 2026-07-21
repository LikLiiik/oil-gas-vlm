class GeoAdapterError(RuntimeError):
    """Base error carrying a user-facing message."""


class ConfigurationError(GeoAdapterError):
    """Raised when input configuration is invalid."""


class OptionalDependencyError(GeoAdapterError):
    """Raised when a requested format needs an unavailable extra."""


class InputDataError(GeoAdapterError):
    """Raised when an input cannot be read safely."""

