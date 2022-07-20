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


def warn_failed_peak_fit(quad, ch):
    return "failed channel {}{:02d} peak fit.".format(quad, ch)


def warn_failed_linearity_fit(quad, ch):
    return "failed channel {}{:02d} linearity fit.".format(quad, ch)


def warn_failed_peak_detection(quad, ch):
    return "failed channel {}{:02d} peak detection".format(quad, ch)
