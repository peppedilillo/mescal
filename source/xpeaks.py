from itertools import combinations
from math import ceil
from math import floor

from lmfit.models import LinearModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats

import source.errors as err


def find_xpeaks(
    bins,
    counts,
    energies,
    gain_guess,
    offset_guess,
    mincounts=100,
    width=5,
    distance=5,
    smoothing=5,
    channel_id=None,
):
    """
    given a histogram of channel counts, a list of spectrum
    energies and a prior on the expected gain and offset
    parameters returns best guess of spectral line positions
    according different criteria

    Args:
        bins: array of int (histograms bin edges)
        counts:  array of int (histogram counts)
        energies: array of floats, energies in keV
        gain_guess: 2-tuple, center and std of gain guess
        offset_guess: 2-tuple, center and std of offset guess
        distance: int, min bins between peaks
        width: int, min peak bins width
        mincounts: int, min counts under peak
        smoothing: int, moving average smoothing parameter
        channel_id: (str, int) tuple, for debugging purpose

    Returns: array of 2-tuples, peaks guess limits indeces.

    """
    assert len(energies) > 1

    initial_search_pars = {
        "prominence": max(counts),
        "width": width,
        "distance": distance,
    }

    # look over the smoothed channel histogram counts for
    # peaks larger than a minimum.
    peaks, peaks_props = peaks_with_enough_stat(
        counts,
        mincounts,
        initial_search_pars,
        smoothing=smoothing,
    )
    # crash and burn if we found no suitable peaks
    if len(peaks) == 0:
        raise err.DetectPeakError("no peaks!")

    peaks, peaks_props = remove_baseline(
        bins,
        counts,
        gain_guess,
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
            for lo, hi in zip(
                best_peaks_props["left_ips"], best_peaks_props["right_ips"]
            )
        ]
        return limits

    # if we have many candidates we consider all the peak combinations
    k = len(energies)
    peaks_combo = np.array([*combinations(peaks, r=k)])
    pcombo_prom = np.array([*combinations(peaks_props["prominences"], r=k)])
    pcombo_widths = np.array([*combinations(peaks_props["widths"], r=k)])

    # we inject some noise into combinations (ordered by default)
    rng = np.random.default_rng()
    permutation = rng.permutation(len(peaks_combo))
    peaks_combo = peaks_combo[permutation]
    pcombo_prom = pcombo_prom[permutation]
    pcombo_widths = pcombo_widths[permutation]

    # for each combination we compute some metrics
    # posterior of peaks position
    posteriorscore_ = posteriorscore(
        bins, peaks_combo, energies, gain_guess, offset_guess
    )
    # peaks linearity
    linscores_, fitparameters = linscores(bins, energies, peaks_combo)
    # peaks prominence
    promscores_ = promscores(pcombo_prom)
    # baseline distance from 2keV
    blscores_ = baselinescores(bins, counts, fitparameters)
    # width coefficient of variation
    widthscores_ = widthscores(bins, gain_guess, pcombo_widths)

    # we rank the combinations by each metric score
    posteriorranking = np.argsort(np.argsort(posteriorscore_))
    linranking = np.argsort(np.argsort(linscores_))
    promranking = np.argsort(np.argsort(promscores_))
    baseranking = np.argsort(np.argsort(blscores_))
    widthranking = np.argsort(np.argsort(widthscores_))

    # best combination is the one with best ranking across metrics
    combined_score = promranking + baseranking + widthranking + posteriorranking
    solve_ties_by = baseranking
    if len(energies) > 2:
        # when we have just two peaks scoring linearity makes no sense
        combined_score += linranking
        solve_ties_by = linscores_
    best_score = max(combined_score)
    # solve ties
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


def widthscores(bins, gain_guess, peaks_combinations_widths):
    """
    evaluates coefficient of variation in peaks width.
    """

    def fano(E):
        """
        Returns Fano noise in keV FWHM
        Input: energy in keV
        """
        return (np.sqrt(E * 1000.0 / 3.6 * 0.118) * 2.35 * 3.6) / 1000

    gain_mean, _ = gain_guess
    widths_bins = bins[peaks_combinations_widths.astype(int)] - bins[0]
    widths_keV = widths_bins / gain_mean
    corrected_widths_keV = fano(widths_keV)
    scores = -np.std(corrected_widths_keV, axis=1) / np.mean(corrected_widths_keV)
    return scores


def baselinescores(bins, counts, fitpars_combinations, thr_energy=2.0):
    """
    evaluates threshold energy given fit parameters of peak combinations.
    """
    baseline = find_baseline(counts)
    scores = [
        -(((bins[baseline] - offset) / gain - thr_energy) ** 2)
        for offset, gain in fitpars_combinations
    ]
    return scores


def promscores(peaks_combinations_proms):
    """
    evaluates prominence of a peak combination.
    """
    scores = [np.prod(peaks_proms) for peaks_proms in peaks_combinations_proms]
    return scores


def linscores(bins, energies, peaks_combinations):
    """
    evaluates the linearity of a peak combination.
    """
    assert np.all(np.diff(energies) > 0)  # is sorted
    assert np.all(np.diff(peaks_combinations) > 0)  # is sorted

    scores = []
    params = []

    for peaks in peaks_combinations:
        lmod = LinearModel()
        pars = lmod.guess(peaks, x=energies)
        resultlin = lmod.fit(
            bins[peaks],
            pars,
            x=energies,
        )
        gain = resultlin.params["slope"].value
        offset = resultlin.params["intercept"].value
        model_predictions = gain * np.array(energies) + offset
        squared_errors = np.square(bins[peaks] - model_predictions)
        u = np.sum(squared_errors)
        v = np.sum((model_predictions - model_predictions.mean()) ** 2)
        scores.append(1 - u / v)
        params.append((offset, gain))
    scores = np.array(scores)
    return scores, params


def posteriorscore(bins, peaks_combinations, energies, gain_guess, offset_guess):
    """
    evaluates peaks combinations given a prior on guess and offset.
    """
    assert np.all(np.diff(energies) > 0)  # is sorted
    assert np.all(np.diff(peaks_combinations) > 0)  # is sorted

    gain_center, gain_sigma = gain_guess
    offset_center, offset_sigma = offset_guess
    mus = gain_center * np.array(energies) + offset_center
    standard_distances = np.abs((bins[peaks_combinations] - mus) / gain_sigma)
    scores = -np.sum(standard_distances, axis=1)
    return scores


def find_baseline(counts):
    """
    low energy threshold index.

    Args:
        counts: array of int

    Returns: an int index

    """
    baseline = np.argwhere((counts[1:] != 0) & (counts[:-1] != 0))[0][0]
    return baseline


def peaks_with_enough_stat(counts, mincounts, pars, smoothing=1, maxdepth=20):
    """
    Iterates scipy.signal.find_peaks() over counts
    until all peaks spans more than mincounts.
    see scipy.signal.find_peaks() for more
    info on peaks and peaks_properties.

    Args:
        counts: array of int, histograms counts
        mincounts: int
        pars: dict, initial peak search parameters
        smoothing: int, window length for SMA smoothing
        maxdepth: maximum iterations

    Returns: peaks and peaks properties

    """
    smooth_counts = moving_average(counts, smoothing)
    peaks, peaks_props = scipy.signal.find_peaks(smooth_counts, **pars)
    for i in range(maxdepth):
        pars["prominence"] = pars["prominence"] / 2
        peaks, peaks_props = scipy.signal.find_peaks(smooth_counts, **pars)
        if not enough_statistics(mincounts, counts, peaks, peaks_props):
            break
    if i == maxdepth - 1:
        raise err.DetectPeakError("reached max depth looking for peaks.")
    if peaks.any():
        peaks, peaks_props = remove_small_peaks(mincounts, counts, peaks, peaks_props)
    return peaks, peaks_props


def moving_average(arr, wlen):
    """
    wrapper to pandas moving average routine.

    Args:
        arr: array of ints (histograms counts)
        wlen: int (window length)

    Returns: array of floats

    """
    return pd.Series(arr).rolling(wlen, center=True).mean().to_numpy()


def remove_small_peaks(mincounts, counts, peaks, peaks_properties):
    """
    remove peaks spanning less than a count minimum
    from peak list and properties.
    see scipy.signal.find_peaks() for more
    info on peaks and peaks_properties.

    Args:
        mincounts: int
        counts: array of ints (histograms counts)
        peaks: array of int (peaks indeces rel. to counts)
        peaks_properties: dictionary of arrays

    Returns: updated peaks and peaks properties

    """
    to_be_removed = []
    for parg, _ in enumerate(peaks):
        if spans_counts(counts, parg, peaks, peaks_properties) < mincounts:
            to_be_removed.append(parg)
    peaks, peaks_properties = _remove(to_be_removed, peaks, peaks_properties)
    return peaks, peaks_properties


def remove_baseline(bins, counts, gain_guess, peaks, peaks_properties, closer_than=1.0):
    """
    Remove peaks close to the baseline.
    Closeness is computed in units of energy,
    relative to a guess on the detector's gain.

    Args:
        bins: array of int, histograms bins
        counts: array of int, histograms counts
        gain_guess: float, a guess on the gain
        peaks: array of int (peaks indeces rel. to counts)
        peaks_properties: dictionary of arrays
        closer_than: float, energy threshold value in keV

    Returns: updated peaks and peaks properties

    """
    gain_center, gain_sigma = gain_guess
    baseline = find_baseline(counts)
    threshold = (gain_center - gain_sigma) * closer_than

    to_be_removed = []
    for parg, _ in enumerate(peaks):
        if bins[peaks[parg]] - bins[baseline] < threshold:
            to_be_removed.append(parg)
    peaks, peaks_properties = _remove(to_be_removed, peaks, peaks_properties)
    return peaks, peaks_properties


def _remove(parg, peaks, peaks_properties):
    """
    remove elements from peaks and peaks_properties
    see scipy.signal.find_peaks() for more
    info on peaks and peaks_properties.

    Args:
        parg: int or array (peaks indeces rel. to peaks)
        peaks: array of int (peaks indeces rel. to counts)
        peaks_properties: dictionary of arrays

    Returns: updated peaks and peaks properties

    """
    mask = np.ones(len(peaks), dtype=bool)
    mask[parg] = False
    peaks = peaks[mask]
    properties = {key: value[mask] for key, value in peaks_properties.items()}
    return peaks, properties


def enough_statistics(mincounts, counts, peaks, peaks_properties):
    """
    return False if at least one peak in peaks
    spans less than a minimum number of counts.
    see scipy.signal.find_peaks() for more
    info on peaks and peaks_properties.

    Args:
        mincounts: int
        counts: array of int (histograms counts)
        peaks: array of int (peaks indeces rel. to counts)
        peaks_properties: dictionary of arrays

    Returns: bool

    """
    if sum(counts) < mincounts:
        return False
    for parg, _ in enumerate(peaks):
        if spans_counts(counts, parg, peaks, peaks_properties) < mincounts:
            return False
    return True


def spans_counts(counts, parg, peaks, peaks_properties):
    """
    counts under a peak (at fwhm).
    see scipy.signal.find_peaks() for more
    info on peaks and peaks_properties.

    Args:
        counts: array of int (histograms counts)
        parg: int (peak index rel. to peaks)
        peaks: array of int (peaks indeces rel. to counts)
        peaks_properties: dictionary of arrays

    Returns: int

    """
    lo = floor(peaks_properties["left_ips"][parg])
    hi = ceil(peaks_properties["right_ips"][parg])
    sumcounts = sum(counts[lo:hi]) - (hi - lo) * min(counts[lo:hi])
    return sumcounts


def debug_helper(
    bins,
    counts,
    peaks_combo,
    labels,
    scores,
    rankings,
    winner,
    winpeaks,
    winpeaks_props,
    peaks,
    peaks_props,
    channel_id,
):
    """
    for debugging purpose

    ex. call:
    debug_helper(
        bins,
        counts,
        peaks_combo,
        ['posterior', 'linearity', 'prominence', 'baseline', 'width'],
        [posteriorscore_, linscores_, promscores_, blscores_, widthscores_],
        [posteriorranking, linranking, promranking, baseranking, widthranking],
        winner,
        best_peaks,
        best_peaks_props,
        peaks,
        peaks_props,
        channel_id,
    )
    """
    quad, ch = channel_id
    print("\n---{}{}---".format(quad, ch))
    print("winner index: ", winner)
    for label, score, ranking in zip(labels, scores, rankings):
        print("{}:".format(label))
        # print(score)
        # print(ranking)
        print(">> winner ranking: {}/{}".format(ranking[winner] + 1, len(peaks_combo)))

    plt.plot(bins[:-1], counts)
    baseline = np.argwhere((counts[1:] != 0) & (counts[:-1] != 0))[0][0]
    plt.axvline(bins[baseline])
    for peak, lo, hi in zip(peaks, peaks_props["left_ips"], peaks_props["right_ips"]):
        plt.axvspan(
            bins[int(lo)],
            bins[int(hi)],
            alpha=0.1,
            color="grey",
        )
    for peak, lo, hi in zip(
        winpeaks, winpeaks_props["left_ips"], winpeaks_props["right_ips"]
    ):
        plt.axvspan(
            bins[int(lo)],
            bins[int(hi)],
            alpha=0.2,
            color="red",
        )
    plt.show()
    plt.close()
