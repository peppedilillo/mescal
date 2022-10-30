from math import ceil, floor
from itertools import combinations
import scipy.signal
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib;
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
            print("stopped at prominence {}".format(pars["prominence"]))
            break
    if i == MAXDEPTH - 1:
        raise TimeoutError("reached max depth looking for peaks")
    if candidate_peaks.any():
        print("candidate peaks: {} peaks".format(len(candidate_peaks)))
        candidate_peaks, candidate_peaks_properties = remove_small_peaks(MINSTAT, counts, candidate_peaks, candidate_peaks_properties)
        print("after small peaks filter: {} peaks".format(len(candidate_peaks)))
        #candidate_peaks, candidate_peaks_properties = remove_noise_corner(candidate_peaks, candidate_peaks_properties, bins, counts)
        print("after noise corner peaks filter: {} peaks".format(len(candidate_peaks)))
    num_ens = len(energies)
    if len(candidate_peaks) < num_ens:
        return False

    candidate_peaks_combinations = np.array([*combinations(candidate_peaks, r=num_ens)])
    candidate_peaks_combinations_leftips = np.array([*combinations(candidate_peaks_properties["left_ips"], r=num_ens)])
    candidate_peaks_combinations_rightips = np.array([*combinations(candidate_peaks_properties["right_ips"], r=num_ens)])
    candidate_peaks_combinations_widths = np.array([*combinations(candidate_peaks_properties["widths"], r=num_ens)])
    candidate_peaks_combinations_proms = np.array([*combinations(candidate_peaks_properties["prominences"], r=num_ens)])

    gain_center, gain_sigma = gain_guess
    offset_center, offset_sigma = offset_guess
    mus = [gain_center * energy + offset_center for energy in energies]
    covmat = [[gain_sigma**2 * energyi * energyj + offset_sigma**2
               for energyj in energies] for energyi in energies]
    logpdfs = scipy.stats.multivariate_normal(mean=mus, cov=covmat, allow_singular=True).logpdf(bins[candidate_peaks_combinations])
    pdfranking = np.argsort(np.argsort(logpdfs))
    print("logpdfs: ", logpdfs)

    model = LinearRegression(fit_intercept=True)
    linscores = []
    params = []
    for peaks, peaks_widths in zip(candidate_peaks_combinations, candidate_peaks_combinations_widths):
        model.fit(np.array(energies).reshape(-1, 1), bins[peaks])
        model_predictions = model.coef_*np.array(energies) + model.intercept_
        squared_errors = np.abs(bins[peaks] - model_predictions)
        #weights = 1#/peaks_widths**2
        u = np.sum(squared_errors)
        v = np.sum((model_predictions - model_predictions.mean())**2)
        linscores.append(1 - u/v)
        params.append((model.intercept_, model.coef_[0]))
    linscores = np.array(linscores)
    linranking = np.argsort(np.argsort(linscores))
    print("linscores: ", linscores)

    promscores = []
    for peaks, peaks_proms in zip(candidate_peaks_combinations, candidate_peaks_combinations_proms):
        promscores.append(np.sum(peaks_proms))
    promranking = np.argsort(np.argsort(promscores))
    print("promscores: ", promscores)

    basescores = []
    baseline = np.argwhere((counts[1:] != 0) & (counts[:-1] != 0))[0][0]
    for peaks, (offset, gain) in zip(candidate_peaks_combinations, params):
        basescores.append(-((bins[baseline] - offset)/gain - 2.0)**2)
    print("basescores: ", basescores)
    baseranking = np.argsort(np.argsort(basescores))


    # noisescore = []
    # maxscore = len(candidate_peaks_combinations)
    # for peaks in candidate_peaks_combinations:
    #     if candidate_peaks[0] in peaks:
    #         noisescore.append(0)
    #     else:
    #         noisescore.append(maxscore)
    # noiseranking = np.argsort(np.argsort(noisescore))
    # print("noisescores: ", noisescore)

    widthscore = []
    for i, (peaks, lipss, ripss) in enumerate(zip(candidate_peaks_combinations, candidate_peaks_combinations_leftips, candidate_peaks_combinations_rightips)):
        diffs = [- ((ripss[n] - lipss[n]) - (ripss[m] - lipss[m]))**2 for n, m in combinations(range(len(peaks)), r= 2)]
        widthscore.append(sum(diffs))
    widthranking = np.argsort(np.argsort(widthscore))
    print("widthscore: ", widthscore)


    if len(candidate_peaks_combinations) == 1:
        logpdfs = np.array([logpdfs])
        linscores = np.array([linscores])
    combined_score = linranking + pdfranking + promranking + widthscore + basescores
    best_score = max(combined_score)
    winners = np.argwhere(combined_score == best_score).T[0]
    # solve ties
    winner = winners[np.argmax(np.array(linscores)[winners])]
    print("winner linscore: ", linranking[winner])
    print("winner logpdfs: ", pdfranking[winner])
    print("winner promrank: ", promranking[winner])
    print("winner promrank: ", baseranking[winner])
    print('intercept:', params[winner][0])
    print('slope:', params[winner][1])

    peaks = candidate_peaks_combinations[winner]
    peaks_args = np.argwhere(np.isin(candidate_peaks, peaks)).T[0]
    peaks_properties = {key: value[peaks_args]
                        for key, value in candidate_peaks_properties.items()}

    plt.plot(bins[:-1], counts)
    baseline = np.argwhere((counts[1:] != 0) & (counts[:-1] != 0))[0][0]
    plt.axvline(bins[baseline])
    for peak, lo, hi in zip(candidate_peaks, candidate_peaks_properties["left_ips"], candidate_peaks_properties["right_ips"]):
        plt.axvspan(bins[int(lo)], bins[int(hi)], alpha=0.2, color='grey')
    for peak, lo, hi in zip(peaks, peaks_properties["left_ips"], peaks_properties["right_ips"]):
        plt.axvspan(bins[int(lo)], bins[int(hi)], alpha=0.2, color='red')
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


def _remove(parg, from_list, list_properties):
    mask = np.ones(len(from_list), dtype=bool)
    mask[parg] = False
    peaks = from_list[mask]
    properties = {key: value[mask] for key, value in list_properties.items()}
    return peaks, properties


def spans_counts(counts, parg, peaks, peaks_properties):
    peak = peaks[parg]
    lo = floor(peaks_properties["left_ips"][parg])
    hi = ceil(peaks_properties["right_ips"][parg])
    sumcounts = sum(counts[lo:hi]) - (hi - lo)*min(counts[lo:hi])
    return sumcounts


def enough_statistics(mincounts, counts, peaks, peaks_properties):
    if sum(counts) < mincounts:
        return False
    for parg, _ in enumerate(peaks):
        if spans_counts(counts, parg, peaks, peaks_properties) < mincounts:
            return False
    return True


def remove_noise_corner(peaks, peaks_properties, bins, counts):
    first_non_zero_bin = np.argwhere((counts[1:] != 0) & (counts[:-1] != 0))[0][0]
    peak, *_ = peaks
    left_ips, *_ = peaks_properties["left_ips"]
    if peak - 5.000/2.355*(peak-left_ips) < first_non_zero_bin:
        return _remove(0, peaks, peaks_properties)
    return peaks, peaks_properties


def remove_small_peaks(mincounts, counts, peaks, peaks_properties):
    to_be_removed = []
    for parg, _ in enumerate(peaks):
        if spans_counts(counts, parg, peaks, peaks_properties) < mincounts:
            to_be_removed.append(parg)
    peaks, peaks_properties = _remove(to_be_removed, peaks, peaks_properties)
    return peaks, peaks_properties
