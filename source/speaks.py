from itertools import combinations
from math import ceil
from math import floor

from lmfit.models import LinearModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import source.errors as err
from source.xpeaks import peaks_with_enough_stat, remove_baseline
from source.constants import PHOTOEL_PER_KEV


def plot(bins, counts, peaks, peaks_props=None, channel_id=None, xlim=None):
    """For debugging purpose."""
    import matplotlib.pyplot as plt

    plt.plot(bins[:-1], counts)
    for peak in peaks:
        plt.axvline(bins[peak])
    if peaks_props:
        for left in peaks_props["left_ips"]:
            plt.axvline(bins[int(left)], ls="dotted")
        for right in peaks_props["right_ips"]:
            plt.axvline(bins[int(right)], ls="dotted")
    if channel_id:
        plt.title(channel_id)
    if xlim:
        plt.xlim(xlim)
    plt.show()


def center_limits(peak, lo_limit, hi_limit):
    width = min(peak - lo_limit, hi_limit + peak)
    return int(floor(peak - width)), int(ceil(peak + width))


def to_bins(bins):
    return lambda n: bins[n]


def find_speaks(
        bins,
        counts,
        energies,
        gain,
        offset,
        lightout_guess,
        width,
        distance,
        smoothing,
        mincounts,
        channel_id=None,
):
    initial_search_pars = {
        "prominence": max(counts),
        "width": width,
        "distance": distance
    }

    peaks, peaks_props = peaks_with_enough_stat(
        counts,
        mincounts,
        initial_search_pars,
        smoothing=smoothing,
    )

    if len(peaks) == 0:
        raise err.DetectPeakError("no peaks!")

    peaks, peaks_props = remove_baseline(
        bins,
        counts,
        gain,
        peaks,
        peaks_props,
    )
    # crash and burn if not enough peaks.
    if len(peaks) < len(energies):
        raise err.DetectPeakError("not enough peaks!")

    # avoid calculations if peak number is right already.
    elif len(peaks) == len(energies):
        best_peaks, best_peaks_props = peaks, peaks_props
        limits = [
            (bins[floor(lo)], bins[ceil(hi)])
            for lo, hi in zip(best_peaks_props["left_ips"], best_peaks_props["right_ips"])
        ]
        return limits

    # if we have many candidates we consider all the peak combinations
    k = len(energies)
    peaks_combo = np.array([*combinations(peaks, r=k)])
    pcombo_prom = np.array([*combinations(peaks_props["prominences"], r=k)])

    # we inject some noise into combinations (ordered by default)
    rng = np.random.default_rng()
    permutation = rng.permutation(len(peaks_combo))

    # computes scores
    posteriorscores_ = posteriorscore(bins, peaks_combo, energies, gain, offset, lightout_guess)

    # computes rankings
    posteriorranking = np.argsort(np.argsort(posteriorscores_))

    # sum rankings. pretty unuseful with just prominence but now we can work on this.
    combined_score = posteriorranking

    # this would result in an error for single energies
    if len(energies) > 1:
        deltascores_ = deltascore(bins, peaks_combo, energies, gain, offset, lightout_guess)
        deltaranking = np.argsort(np.argsort(deltascores_))
        combined_score += deltaranking

    # determines winner solving possible ties
    solve_ties_by = posteriorranking
    best_score = max(combined_score)
    tied_for_win = np.argwhere(combined_score == best_score).T[0]
    winner = tied_for_win[np.argmax(np.array(solve_ties_by)[tied_for_win])]

    # some packaging
    best_peaks = peaks_combo[winner]
    best_peaks_args = np.argwhere(np.isin(peaks, best_peaks)).T[0]
    best_peaks_props = {
        key: value[best_peaks_args] for key, value in peaks_props.items()
    }

    limits = [
        (bins[floor(lo)], bins[ceil(hi)])
        for lo, hi in zip(best_peaks_props["left_ips"], best_peaks_props["right_ips"])
    ]
    return limits


def promscores(peaks_combinations_proms):
    """
    evaluates prominence of a peak combination.
    """
    scores = [np.sum(peaks_proms) for peaks_proms in peaks_combinations_proms]
    # print(
    #     "\n\nPROM SCORE",
    #     "\nscores:", scores)
    return scores


def posteriorscore(bins, peaks_combinations, energies, gain, offset, lightout_guess):
    """
    evaluates peaks combinations given a prior on gain, offset, light output.
    """
    assert np.all(np.diff(energies) > 0)  # is sorted
    assert np.all(np.diff(peaks_combinations) > 0)  # is sorted

    gain_center, gain_sigma = gain
    offset_center, offset_sigma = offset
    bins_kev = (bins - offset_center) / gain_center
    lightout_center, lightout_sigma = lightout_guess
    # the halving is because we will deal with one cell
    energies_equivalent_kev = PHOTOEL_PER_KEV * lightout_center / 2 * np.array(energies)
    standard_distances = np.abs(bins_kev[peaks_combinations] - energies_equivalent_kev)
    scores = -np.sum(standard_distances, axis=1)
    return scores


def deltascore(bins, peaks_combinations, energies, gain, offset, lightout_guess):
    """
    evaluates peaks distance given a prior on guess and offset.
    """
    assert np.all(np.diff(energies) > 0)  # is sorted
    assert np.all(np.diff(peaks_combinations) > 0)  # is sorted

    gain_center, gain_sigma = gain
    offset_center, offset_sigma = offset
    bins_kev = (bins - offset_center) / gain_center
    lightout_center, lightout_sigma = lightout_guess
    energies_equivalent_kev = PHOTOEL_PER_KEV * lightout_center / 2 * np.array(energies)
    deltaen_true_kev = np.diff(energies_equivalent_kev)
    deltaen_observed_kev = np.diff(bins_kev[peaks_combinations])
    scores = -np.sum(np.square(np.diff(deltaen_observed_kev - deltaen_true_kev)))
    return scores


PROMINENCE_WEIGHTING = False


def find_epeaks(
        bins,
        counts,
        sfit_ch,
        sfit_comp,
        gain_ch,
        gain_comp,
        offset_ch,
        offset_comp,
        mincounts,
        width,
        smoothing,
        distance,
        channel_id=None,
):
    gain_ch_center, _ = gain_ch
    gain_comp_center, _ = gain_comp
    offset_ch_center, _ = offset_ch
    offset_comp_center, _ = offset_comp
    pe_ch = (sfit_ch - offset_ch_center) / gain_ch_center / PHOTOEL_PER_KEV
    pe_comp = (sfit_comp - offset_comp_center) / gain_comp_center / PHOTOEL_PER_KEV
    pe_guess = pe_ch + pe_comp

    initial_search_pars = {
        "prominence": max(counts),
        "width": width,
        "distance": distance
    }
    peaks, peaks_props = peaks_with_enough_stat(
        counts,
        mincounts,
        initial_search_pars,
        smoothing=smoothing,
    )

    if len(peaks) >= len(pe_guess):
        best_peaks, best_peaks_props = _closest_peaks(pe_guess, bins, peaks, peaks_props)
    else:
        raise err.DetectPeakError("candidate peaks are less than sources to fit.")

    limits = [
        (bins[floor(lo)], bins[ceil(hi)])
        for lo, hi in zip(best_peaks_props["left_ips"], best_peaks_props["right_ips"])
    ]
    return limits


def guess_epeaks_position(energies, lightout_guess):
    center, sigma = lightout_guess
    lightout_lims = center + sigma, center - sigma
    guesses = [[lo * lv for lo in lightout_lims] for lv in energies]
    return guesses


def moving_average(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()


def _dist_from_intv(x, lo, hi):
    return abs((x - lo) + (x - hi))


def _closest_peaks(guess, bins, peaks, peaks_infos):
    peaks_dist_from_guess = np.array(
        [
            [abs(guess - bins[peak]) for peak in peaks]
            for guess in guess
        ]
    )
    args_sorted_distances = np.argsort(np.min(peaks_dist_from_guess, axis=1))
    args_peaks = []
    for arg in args_sorted_distances:
        line = peaks_dist_from_guess[arg]
        arg_closest_peak = np.argmin(line)
        args_peaks.append(arg_closest_peak)
        peaks_dist_from_guess[:, arg_closest_peak] = np.inf
    argmin = np.sort(args_peaks)

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
