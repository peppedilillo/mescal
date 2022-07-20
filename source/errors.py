class ModelNotFoundError(Exception):
    """An error when querying an unsupported detector."""


class FormatNotSupportedError(Exception):
    """An error while asking for unsupported writing formats."""


class SourceNotFoundError(Exception):
    """An error while parsing calib sources."""


class DetectPeakError(Exception):
    """An error while finding peaks."""


class PoorFitError(Exception):
    """An when trying to fit a bad peak."""


class FailedFitError(Exception):
    """An error when fitting fails."""


class FailedCalibrationError(Exception):
    """An error when fitting fails."""
