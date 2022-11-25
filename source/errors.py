class DetectorModelNotFound(Exception):
    """An error when querying an unsupported detector."""


class FormatNotSupportedError(Exception):
    """An error while asking for unsupported writing formats."""


class SourceNotFoundError(Exception):
    """An error while parsing calib sources."""


class DefaultCalibNotFoundError(Exception):
    """An error when missing default calibration."""


class FewLinesError(Exception):
    """Badly constrained sources to fit."""


class DetectPeakError(Exception):
    """An error while finding peaks."""


class FailedFitError(Exception):
    """An error when fitting fails."""


class CalibratedEventlistError(Exception):
    """An error when building calibrated event list."""


def warn_peak_fit(quad, ch):
    return "failed channel {}{:02d} peak fit.".format(quad, ch)


def warn_failed_linearity_fit(quad, ch):
    return "failed channel {}{:02d} linearity fit.".format(quad, ch)


def warn_peak_detection(quad, ch):
    return "failed channel {}{:02d} peak detection".format(quad, ch)


def warn_peak_detection_offcomp(quad, ch, companion):
    return (
        "failed peak detection of {0}{1:02d}, companion cell {0}{2:02d}"
        " appears to be off.".format(quad, ch, companion)
    )


def warn_missing_defcal(quad, ch):
    return "missing default calibration for channel {}{:02d}.".format(quad, ch)


def warn_missing_lout(quad, ch):
    return "missing light output for channel {}{:02d}".format(quad, ch)
