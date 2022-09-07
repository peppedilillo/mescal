import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import source.errors as err

PROMINENCE_WEIGHTING = True

SMOOTHING = 5

PEAKS_DETECTION_PARAMETERS = {
    "prominence": 70,
    "width": 5,
    "distance": 10,
}


def move_mean(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()


def find_xlimits(
    bins,
    counts,
    radsources: list,
    gain_guess=None,
    offset_guess=None,
):
    if (gain_guess is not None) and (offset_guess is not None):
        return _lims_from_guess(bins, counts, radsources, gain_guess, offset_guess)
    elif (gain_guess is not None) or (offset_guess is not None):
        raise ValueError("not yet implemented!")
    else:
        return _lims_from_decays_ratio(bins, counts, radsources)


def _lims_from_guess(
    bins,
    counts,
    radsources: list,
    gain_guess,
    offset_guess,
    find_peaks_params=None,
):
    if find_peaks_params is None:
        find_peaks_params = PEAKS_DETECTION_PARAMETERS

    low_en_threshold = 1.0  # keV

    energies = (bins - offset_guess) / gain_guess
    ((inf_bin, *_),) = np.where(energies > low_en_threshold)
    smoothed_counts = move_mean(counts, SMOOTHING)
    unfiltered_peaks, unfiltered_peaks_info = find_peaks(
        smoothed_counts,
        **find_peaks_params,
    )
    enfiltered_peaks, enfiltered_peaks_info = _filter_peaks_low_energy(
        inf_bin,
        unfiltered_peaks,
        unfiltered_peaks_info,
    )
    if len(enfiltered_peaks) < len(radsources):
        raise err.DetectPeakError("candidate peaks are less than radsources to fit.")
    peaks, peaks_info = _filter_peaks_proximity(
        radsources,
        energies,
        enfiltered_peaks,
        enfiltered_peaks_info,
    )
    limits = [
        (bins[int(p - w)], bins[int(p + w)])
        for p, w in zip(peaks, peaks_info["widths"])
    ]
    return limits


def _filter_peaks_proximity(radsources: list, energies, peaks, peaks_infos):
    peaks_combinations = [*combinations(peaks, r=len(radsources))]
    enpeaks_combinations = np.take(energies, peaks_combinations)
    loss = np.sum(np.square(enpeaks_combinations - np.array(radsources)), axis=1)
    filtered_peaks = peaks_combinations[np.argmin(loss)]
    filtered_peaks_info = {
        key: val[np.isin(peaks, filtered_peaks)] for key, val in peaks_infos.items()
    }
    return filtered_peaks, filtered_peaks_info


def _filter_peaks_low_energy(lim_bin, peaks, peaks_infos):
    filtered_peaks = peaks[np.where(peaks > lim_bin)]
    filtered_peaks_info = {
        key: val[np.isin(peaks, filtered_peaks)] for key, val in peaks_infos.items()
    }
    return filtered_peaks, filtered_peaks_info


def _lims_from_decays_ratio(
    bins,
    counts,
    radsources: list,
    find_peaks_params=None,
):
    if find_peaks_params is None:
        find_peaks_params = PEAKS_DETECTION_PARAMETERS

    if len(radsources) < 3:
        raise err.DetectPeakError("not enough radsources to calibrate.")

    mm = move_mean(counts, SMOOTHING)
    unfiltered_peaks, unfiltered_peaks_info = find_peaks(
        mm,
        **find_peaks_params,
    )
    if len(unfiltered_peaks) < len(radsources):
        raise err.DetectPeakError("candidate peaks are less than radsources to fit.")

    peaks, peaks_info = _filter_peaks_lratio(
        radsources,
        unfiltered_peaks,
        unfiltered_peaks_info,
    )
    limits = [
        (bins[int(p - w)], bins[int(p + w)])
        for p, w in zip(peaks, peaks_info["widths"])
    ]
    return limits


def _normalize(x):
    return [(x[i + 1] - x[i]) / (x[-1] - x[0]) for i in range(len(x) - 1)]


def _weight(x):
    return [x[i + 1] * x[i] for i in range(len(x) - 1)]


def _filter_peaks_lratio(radsources: list, peaks, peaks_infos):
    peaks_combinations = [*combinations(peaks, r=len(radsources))]
    proms_combinations = combinations(peaks_infos["prominences"], r=len(radsources))
    norm_ls = _normalize(radsources)
    norm_ps = [*map(_normalize, peaks_combinations)]
    if PROMINENCE_WEIGHTING:
        weights = [*map(_weight, proms_combinations)]
        loss = np.sum(
            np.square(np.array(norm_ps) - np.array(norm_ls)) / np.square(weights),
            axis=1,
        )
    else:
        loss = np.sum(np.square(np.array(norm_ps) - np.array(norm_ls)), axis=1)
    best_peaks = peaks_combinations[np.argmin(loss)]
    best_peaks_info = {
        key: val[np.isin(peaks, best_peaks)] for key, val in peaks_infos.items()
    }
    return best_peaks, best_peaks_info
