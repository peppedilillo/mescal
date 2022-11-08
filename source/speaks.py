import numpy as np
import pandas as pd
from math import floor, ceil
from scipy.signal import find_peaks

from source.constants import PHOTOEL_PER_KEV
from source.errors import DetectPeakError

SMOOTHING = 20

SPEAKS_DETECTION_PARAMETERS = {
    "prominence": 6,
    "width": 10,
}


EPEAKS_DETECTION_PARAMETERS = {
    "prominence": 30,
    "width": 10,
}

PROMINENCE_WEIGHTING = False


def find_epeaks(bins, counts, energies, lightout_guess, smoothing=20, prominence=30, width=10):
    search_pars = {
        "prominence": prominence,
        "width": width,
    }
    mm = moving_average(counts, smoothing)
    peaks, peaks_props = find_peaks(mm, **search_pars)
    guesses = guess_epeaks_position(energies, lightout_guess)
    if len(peaks) >= len(guesses):
        best_peaks, best_peaks_props = _closest_peaks(guesses, bins, peaks, peaks_props)
    else:
        raise DetectPeakError("candidate peaks are less than sources to fit.")
    limits = [
        (bins[int(p - w)], bins[int(p + w)])
        for p, w in zip(best_peaks, best_peaks_props["widths"])
    ]
    return limits


def guess_epeaks_position(energies, lightout_guess):
    center, sigma = lightout_guess
    lightout_lims = center + sigma, center - sigma
    guesses = [[lo * lv for lo in lightout_lims] for lv in energies]
    return guesses


def find_speaks(bins, counts, energies, gain, offset, lightout_guess, smoothing=20, prominence=6, width=10):
    search_pars = {
        "prominence": prominence,
        "width": width,
    }
    mm = moving_average(counts, smoothing)
    peaks, peaks_props = find_peaks(mm, **search_pars)
    guesses = guess_speaks_position(energies, lightout_guess, gain, offset)
    if len(peaks) >= len(guesses):
        best_peaks, best_peaks_props = _closest_peaks(guesses, bins, peaks, peaks_props)
    else:
        raise DetectPeakError("candidate peaks are less than sources to fit.")
    limits = [
        (bins[int(p - w)], bins[int(p + w)])
        for p, w in zip(best_peaks, best_peaks_props["widths"])
    ]
    return limits


def guess_speaks_position(energies, lightout_guess, gain, offset):
    center, sigma = lightout_guess
    lightout_lims = center + sigma, center - sigma
    guesses = [
        [
            (0.5 * lout_lim) * PHOTOEL_PER_KEV * lv * gain + offset
            for lout_lim in lightout_lims
        ]
        for lv in energies
    ]
    return guesses


def moving_average(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()


def _dist_from_intv(x, lo, hi):
    return abs((x - lo) + (x - hi))


def _closest_peaks(guess, bins, peaks, peaks_infos):
    peaks_dist_from_guess = np.array([
        [_dist_from_intv(bins[peak], guess_lo, guess_hi) for peak in peaks]
        for guess_lo, guess_hi in guess
    ])
    if PROMINENCE_WEIGHTING:
        assert "prominences" in peaks_infos.keys()
        weights = peaks_infos["prominences"]
        argmin = np.argmin(peaks_dist_from_guess**2/weights, axis=1)
    else:
        argmin = np.argmin(peaks_dist_from_guess, axis=1)
    best_peaks = peaks[argmin]
    best_peaks_infos = {key: val[argmin] for key, val in peaks_infos.items()}
    return best_peaks, best_peaks_infos


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

