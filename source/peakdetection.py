from math import ceil, floor
from itertools import combinations
import scipy.signal
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import source.errors as err


MINCOUNTS = 200
SMOOTHING_LEN = 5
MAXDEPTH = 20


def find_xpeaks(bins, counts, energies, gain_guess, offset_guess):
    # look over the smoothed channel histogram counts for
    # peaks larger than a minimum.
    initial_search_pars = {
        'prominence': sum(counts),
        'width': 5,
        'distance': 5,
    }
    peaks, peaks_props = peaks_with_enough_stat(counts, MINCOUNTS, initial_search_pars)

    # crash and burn if not enough peaks.
    # avoid calculations if peak number is right already.
    if len(peaks) < len(energies):
        raise err.DetectPeakError("not enough peaks!")
    elif len(peaks) == len(energies):
        return peaks

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
    pdfscores_ = pdfscores(bins, peaks_combo, energies, gain_guess, offset_guess)
    linscores_, fitparameters = linscores(bins, energies, peaks_combo)
    promscores_ = promscores(pcombo_prom)
    blscores_ = baselinescores(bins, counts, fitparameters)
    widthscores_ = widthscores(peaks_combo, pcombo_widths)

    # we rank the combinations by each metric score
    pdfranking = np.argsort(np.argsort(pdfscores_))
    linranking = np.argsort(np.argsort(linscores_))
    promranking = np.argsort(np.argsort(promscores_))
    baseranking = np.argsort(np.argsort(blscores_))
    widthranking = np.argsort(np.argsort(widthscores_))

    # best combination is the one with best ranking across metrics
    combined_score = linranking + pdfranking + promranking + widthranking + baseranking
    best_score = max(combined_score)
    # solve ties by linearity
    tied_for_win = np.argwhere(combined_score == best_score).T[0]
    winner = tied_for_win[np.argmax(np.array(linscores_)[tied_for_win])]

    best_peaks = peaks_combo[winner]
    best_peaks_args = np.argwhere(np.isin(peaks, best_peaks)).T[0]
    best_peaks_props = {key: value[best_peaks_args]
                        for key, value in peaks_props.items()}

    debug_helper(
        bins,
        counts,
        peaks_combo,
        ['pdf', 'linearity', 'prominence', 'baseline', 'width'],
        [pdfscores_, linscores_, promscores_, blscores_, widthscores_],
        [pdfranking, linranking, promranking, baseranking, widthranking],
        winner,
        best_peaks,
        best_peaks_props,
        peaks,
        peaks_props)
    return best_peaks


def widthscores(peaks_combinations, peaks_combinations_widths):
    scores = []
    i = 0
    for peaks, widths in zip(peaks_combinations, peaks_combinations_widths):
        print(i, peaks, widths, np.std(widths)/np.mean(widths))
        i+=1
        scores.append(- np.std(widths)/np.mean(widths))
    return scores


def baselinescores(bins, counts, fitpars_combinations):
    baseline = find_baseline(counts)
    scores = [-((bins[baseline] - offset)/gain - 2.0)**2
              for offset, gain in fitpars_combinations]
    return scores


def promscores(peaks_combinations_proms):
    scores = [np.sum(peaks_proms) for peaks_proms in peaks_combinations_proms]
    return scores


def linscores(bins, energies, peaks_combinations):
    model = LinearRegression(fit_intercept=True)
    scores = []
    params = []
    for peaks in peaks_combinations:
        model.fit(np.array(energies).reshape(-1, 1), bins[peaks])
        model_predictions = model.coef_*np.array(energies) + model.intercept_
        squared_errors = np.abs(bins[peaks] - model_predictions)
        u = np.sum(squared_errors)
        v = np.sum((model_predictions - model_predictions.mean())**2)
        scores.append(1 - u/v)
        offset, gain = model.intercept_, model.coef_[0]
        params.append((offset, gain))
    scores = np.array(scores)
    return scores, params


def pdfscores(bins, peaks_combinations, energies, gain_guess, offset_guess):
    gain_center, gain_sigma = gain_guess
    offset_center, offset_sigma = offset_guess
    mus = [gain_center * energy + offset_center for energy in energies]
    covmat = [[gain_sigma**2 * energyi * energyj + offset_sigma**2
               for energyj in energies] for energyi in energies]
    scores = scipy.stats.multivariate_normal(mean=mus, cov=covmat, allow_singular=True).logpdf(bins[peaks_combinations])
    return scores


def find_baseline(counts):
    baseline = np.argwhere((counts[1:] != 0) & (counts[:-1] != 0))[0][0]
    return baseline


def peaks_with_enough_stat(counts, mincounts, pars):
    """
    Iterates scipy.signal.find_peaks() over counts
    until all peaks spans more than mincounts.
    see scipy.signal.find_peaks() for more
    info on peaks and peaks_properties.

    Args:
        counts: array of int
        mincounts: int
        pars: dict

    Returns:

    """
    smooth_counts = moving_average(counts, SMOOTHING_LEN)
    candidate_peaks, candidate_peaks_properties = scipy.signal.find_peaks(smooth_counts, **pars)
    for i in range(MAXDEPTH):
        pars["prominence"] = pars["prominence"]/2
        candidate_peaks, candidate_peaks_properties = scipy.signal.find_peaks(smooth_counts, **pars)
        if not enough_statistics(mincounts, counts, candidate_peaks, candidate_peaks_properties):
            # print("stopped at prominence {}".format(pars["prominence"]))
            break
    if i == MAXDEPTH - 1:
        raise TimeoutError("reached max depth looking for peaks")
    if candidate_peaks.any():
        # print("candidate peaks: {} peaks".format(len(candidate_peaks)))
        candidate_peaks, candidate_peaks_properties = remove_small_peaks(mincounts, counts, candidate_peaks, candidate_peaks_properties)
        # print("after small peaks filter: {} peaks".format(len(candidate_peaks)))
    return candidate_peaks, candidate_peaks_properties
 

def moving_average(arr, wlen):
    """
    wrapper to pandas moving average routine.
    
    Args:
        arr: array of ints (histograms counts)
        wlen: int (window length)

    Returns:

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
    peak = peaks[parg]
    lo = floor(peaks_properties["left_ips"][parg])
    hi = ceil(peaks_properties["right_ips"][parg])
    sumcounts = sum(counts[lo:hi]) - (hi - lo)*min(counts[lo:hi])
    return sumcounts


def _remove(parg, peaks, peaks_properties):
    """
    remove elements from peaks and peaks_properties
    see see scipy.signal.find_peaks() for more
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


def debug_helper(bins, counts, peaks_combo, labels, scores, rankings, winner, winpeaks, winpeaks_props, peaks, peaks_props):
    print("winner index: ", winner)
    for label, score, ranking in zip(labels, scores, rankings):
        print("\n{}:".format(label))
        #print(score)
        #print(ranking)
        print(">> winner ranking: ", ranking[winner])

    plt.plot(bins[:-1], counts)
    baseline = np.argwhere((counts[1:] != 0) & (counts[:-1] != 0))[0][0]
    plt.axvline(bins[baseline])
    for peak, lo, hi in zip(peaks, peaks_props["left_ips"], peaks_props["right_ips"]):
        plt.axvspan(bins[int(lo)], bins[int(hi)], alpha=0.2, color='grey')
    for peak, lo, hi in zip(winpeaks, winpeaks_props["left_ips"], winpeaks_props["right_ips"]):
        plt.axvspan(bins[int(lo)], bins[int(hi)], alpha=0.2, color='red')
    plt.show()