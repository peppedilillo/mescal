class UnknownModelError(Exception):
    """An error when querying an unsupported detector."""


class DetectPeakError(Exception):
    """An error while finding peaks."""


class FormatNotSupportedError(Exception):
    """An error while asking for unsupported writing formats."""


class SourceNotFoundError(Exception):
    """An error while parsing calib sources."""


class CalibrationNotFoundError(Exception):
    """An error when querying for unavailable default calibrations."""
