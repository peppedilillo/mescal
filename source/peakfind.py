from math import ceil, floor
from itertools import combinations
import scipy.signal
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib;
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

MINSTAT = 200
SMOOTHING = 5
MAXDEPTH = 20


def find_peaks(bins, counts, energies, gain_guess, offset_guess):
    smooth_counts = moving_average(counts, SMOOTHING)

    # look for a set of candidate peaks in data.
    # we try to find all the peaks which spans
    # a number of events greater than MINSTAT
    pars = initial_parameters(bins, counts)
    candidate_peaks, candidate_peaks_properties = scipy.signal.find_peaks(smooth_counts, **pars)
    for i in range(MAXDEPTH):
        pars["prominence"]=pars["prominence"]/2
        candidate_peaks, candidate_peaks_properties = scipy.signal.find_peaks(smooth_counts, **pars)
        if not enough_statistics(MINSTAT, counts, candidate_peaks, candidate_peaks_properties):
            break
    if i == MAXDEPTH - 1:
        raise TimeoutError("reached max depth looking for peaks")
    if candidate_peaks.any():
        candidate_peaks, candidate_peaks_properties = remove_small_peaks(MINSTAT, counts, candidate_peaks, candidate_peaks_properties)
        candidate_peaks, candidate_peaks_properties = remove_noise_corner(candidate_peaks, candidate_peaks_properties, bins, counts)


    num_ens = len(energies)
    if len(candidate_peaks) < num_ens:
        return False
    candidate_peaks_combinations = np.array([*combinations(candidate_peaks, r=num_ens)])

    gain_center, gain_sigma = gain_guess
    offset_center, offset_sigma = offset_guess
    mus = [gain_center * energy + offset_center for energy in energies]
    covmat = [[gain_sigma**2 * energyi * energyj + offset_sigma**2
               for energyj in energies] for energyi in energies]
    logpdfs = scipy.stats.multivariate_normal(mean=mus, cov=covmat, allow_singular=True).logpdf(bins[candidate_peaks_combinations])
    pdfranking = np.argsort(np.argsort(logpdfs))
    print("pdfsort: ", np.round(logpdfs, 3), pdfranking)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True)
    linearity = []
    params = []
    for peaks in candidate_peaks_combinations:
        model.fit(np.array(energies).reshape(-1, 1), bins[peaks])
        s = model.score(np.array(energies).reshape(-1, 1), bins[peaks])
        linearity.append(s)
        params.append((model.intercept_, model.coef_))
    linranking = np.argsort(np.argsort(linearity))
    print("linearitysort: ", np.round(linearity, 5), linranking)


    combined_score = linranking
    best_score = max(combined_score)
    winners = np.argwhere(combined_score == best_score).T[0]
    # solve ties
    winner = winners[np.argmax(np.array(linearity)[winners])]
    print('intercept:', params[winner][0])
    print('slope:', params[winner][1])
    peaks = candidate_peaks_combinations[winner]
    peaks_args = np.argwhere(np.isin(candidate_peaks, peaks)).T[0]
    peaks_properties = {key: value[peaks_args]
                        for key, value in candidate_peaks_properties.items()}


    plt.plot(bins[:-1], counts)
    for mu in mus:
        plt.axvline(mu, linestyle='dotted')
    for peak, width in zip(peaks, peaks_properties["widths"]):
        lo, hi = floor(peak - width), ceil(peak + width)
        plt.axvspan(bins[lo], bins[hi], alpha=0.2, color='r')
    for peak, width in zip(candidate_peaks, candidate_peaks_properties["widths"]):
        lo, hi = floor(peak - width), ceil(peak + width)
        plt.axvspan(bins[lo], bins[hi], alpha=0.2, color='grey')
    plt.show()
    return True


def moving_average(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()


def initial_parameters(bins, counts):
    parameters = {
        'prominence': sum(counts),
        'width': 5,
        'distance': 5,
    }
    return parameters


def remove(peak, from_list, list_properties):
    mask = (from_list != peak)
    peaks = from_list[mask]
    properties = {key: value[mask] for key, value in list_properties.items()}
    return peaks, properties


def spans_counts(peak, width, counts):
    lo, hi = floor(peak - width), ceil(peak + width) + 1
    sumcounts = sum(counts[lo:hi]) - 0.5 * ((hi - lo) * (counts[lo] + counts[hi]))
    return sumcounts


def enough_statistics(mincounts, counts, peaks, peaks_properties):
    if sum(counts) < mincounts:
        return False
    for peak, width in zip(peaks, peaks_properties["widths"]):
        if spans_counts(peak, width, counts) < mincounts:
            return False
    return True


def remove_noise_corner(peaks, peaks_properties, bins, counts):
    first_non_zero_bin = np.argwhere(counts > 0).T[0][0]
    peak, *_ = peaks
    width, *_ = peaks_properties["widths"]
    if peak - width  - 1< first_non_zero_bin:
        return remove(peak, peaks, peaks_properties)
    return peaks, peaks_properties


def remove_small_peaks(mincounts, counts, peaks, peaks_properties):
    to_be_removed = []
    for peak, width in zip(peaks, peaks_properties["widths"]):
        if spans_counts(peak, width, counts) < mincounts:
            to_be_removed.append(peak)
    for peak in to_be_removed:
        peaks, peaks_properties = remove(peak, peaks, peaks_properties)
    return peaks, peaks_properties
