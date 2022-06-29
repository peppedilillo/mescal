class UnknownModelError(Exception):
    """An error when querying an unsupported detector."""


class DetectPeakError(Exception):
    """An error while finding peaks."""
