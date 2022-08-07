import numpy as np
from scipy.signal import find_peaks

from source.constants import PHOTOEL_PER_KEV
from source.errors import DetectPeakError
from source.specutils import move_mean

SMOOTHING = 20

PEAKS_DETECTION_PARAMETERS = {
    "prominence": 5,
    "width": 20,
}


def _dist_from_intv(x, lo, hi):
    return abs((x - lo) + (x - hi))


def _closest_peaks(guess, peaks, peaks_infos):
    peaks_dist_from_guess = [
        [_dist_from_intv(peak, guess_lo, guess_hi) for peak in peaks]
        for guess_lo, guess_hi in guess
    ]
    argmin = np.argmin(peaks_dist_from_guess, axis=1)
    best_peaks = peaks[argmin]
    best_peaks_infos = {key: val[argmin] for key, val in peaks_infos.items()}
    return best_peaks, best_peaks_infos


def _estimate_peaks_from_guess(bins, counts, guess, find_peaks_params=None):
    if find_peaks_params is None:
        find_peaks_params = PEAKS_DETECTION_PARAMETERS

    mm = move_mean(counts, SMOOTHING)
    many_peaks, many_peaks_info = find_peaks(mm, **find_peaks_params)
    if len(many_peaks) >= len(guess):
        peaks, peaks_info = _closest_peaks(guess, many_peaks, many_peaks_info)
    else:
        raise DetectPeakError("candidate peaks are less than sources to fit.")
    limits = [
        (bins[int(p - w)], bins[int(p + w)])
        for p, w in zip(peaks, peaks_info["widths"])
    ]
    return limits


def _compute_louts(
    centers, center_errs, gain, gain_err, offset, offset_err, radsources: list
):
    light_outs = (centers - offset) / gain / PHOTOEL_PER_KEV / radsources
    light_out_errs = (
        np.sqrt(
            (center_errs / gain) ** 2
            + (offset_err / gain) ** 2
            + ((centers - offset) / gain**2) * (gain_err**2)
        )
        / PHOTOEL_PER_KEV
        / radsources
    )
    return light_outs, light_out_errs
