import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from source.errors import DetectPeakError
from source.errors import FailedFitError
from source.specutils import fit_radsources_peaks
from source.specutils import move_mean
from source.errors import warn_failed_peak_detection
from source.errors import warn_failed_peak_fit
from source.specutils import PHT_KEV
import logging

FIT_PARAMS = [
    "center",
    "center_err",
    "fwhm",
    "fwhm_err",
    "amp",
    "amp_err",
    "lim_low",
    "lim_high",
]

LOUT_GUESS = (10., 15.)

SMOOTHING = 20

PEAKS_DETECTION_PARAMETERS = {
    'prominence': 5,
    'width': 20,
}


def as_dict_of_dataframes(f):
    def wrapper(*args):
        nested_dict, radsources, flagged = f(*args)
        quadrants = nested_dict.keys()
        index = pd.MultiIndex.from_product(
            (radsources.keys(), FIT_PARAMS),
            names=['source', 'parameter']
        )

        dict_of_dfs = {
            q: pd.DataFrame(
                nested_dict[q],
                index=index
            ).T.rename_axis("channel")
            for q in quadrants
        }
        return dict_of_dfs, flagged
    return wrapper


@as_dict_of_dataframes
def fit_sumhistograms(histograms, radsources, lout_guess=LOUT_GUESS):
    results, flagged = {}, {}
    energies = [s.energy for s in radsources.values()]

    bins = histograms.bins
    for quad in histograms.counts.keys():
        for ch in histograms.counts[quad].keys():
            counts = histograms.counts[quad][ch]

            guesses = [[2 * lout_lim * lv
                        for lout_lim in lout_guess]
                           for lv in energies]

            try:
                limits = _estimate_peaks_from_guess(
                    bins,
                    counts,
                    guess=guesses,
                )
            except DetectPeakError:
                message = warn_failed_peak_detection(quad, ch)
                logging.warning(message)
                flagged.setdefault(quad, []).append(ch)
                continue

            try:
                intervals, fit_results = fit_radsources_peaks(
                    bins,
                    counts,
                    limits,
                    radsources,
                )
            except FailedFitError:
                message = warn_failed_peak_fit(quad, ch)
                logging.warning(message)
                flagged.setdefault(quad, []).append(ch)
                continue

            int_inf, int_sup = zip(*intervals)
            results.setdefault(quad, {})[ch] = np.column_stack(
                (*fit_results, int_inf, int_sup)).flatten()
    return results, radsources, flagged


@as_dict_of_dataframes
def fit_sradsources(histograms, cal_df, radsources, lout_guess=LOUT_GUESS):
    results, flagged = {}, {}
    energies = [s.energy for s in radsources.values()]

    bins = histograms.bins
    for quad in cal_df.keys():
        for ch in cal_df[quad].index:
            counts = histograms.counts[quad][ch]
            gain = cal_df[quad].loc[ch]['gain']
            offset = cal_df[quad].loc[ch]['offset']

            guesses = [[lout_lim * PHT_KEV * lv * gain + offset
                        for lout_lim in lout_guess]
                       for lv in energies]

            try:
                limits = _estimate_peaks_from_guess(
                    bins,
                    counts,
                    guess=guesses,
                )
            except DetectPeakError:
                message = warn_failed_peak_detection(quad, ch)
                logging.warning(message)
                flagged.setdefault(quad, []).append(ch)
                continue

            try:
                intervals, fit_results = fit_radsources_peaks(
                    bins,
                    counts,
                    limits,
                    radsources,
                )
            except FailedFitError:
                message = warn_failed_peak_fit(quad, ch)
                logging.warning(message)
                flagged.setdefault(quad, []).append(ch)
                continue

            int_inf, int_sup = zip(*intervals)
            results.setdefault(quad, {})[ch] = np.column_stack(
                (*fit_results, int_inf, int_sup)).flatten()
    return results, radsources, flagged


def _dist_from_intv(x, lo, hi): return abs((x - lo) + (x - hi))


def _closest_peaks(guess, peaks, peaks_infos):
    peaks_dist_from_guess = [[_dist_from_intv(peak, guess_lo, guess_hi)
                              for peak in peaks]
                             for guess_lo, guess_hi in guess]
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
    limits = [(bins[int(p - w)], bins[int(p + w)])
              for p, w in zip(peaks, peaks_info['widths'])]
    return limits


def _compute_louts(centers, center_errs, gain, gain_err, offset, offset_err, radsources: list):
    light_outs = (centers - offset) / gain / PHT_KEV / radsources
    light_out_errs = np.sqrt((center_errs / gain) ** 2
                             + (offset_err / gain) ** 2
                             + ((centers - offset) / gain ** 2) * (gain_err ** 2)) / PHT_KEV / radsources
    return light_outs, light_out_errs
