from itertools import combinations

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from lmfit.models import GaussianModel, LinearModel, PolynomialModel
from scipy.signal import find_peaks

PHT_KEV = 3.65 / 1000


class DetectPeakError(Exception):
    """An error while finding peaks."""


dist_from_intv = (lambda x, lo, hi: abs((x - lo) + (x - hi)))


def scalibrate(bins, histograms, cal_df, lines, lout_guess):
    results_lo, flagged = {}, {}
    line_keys, line_values = zip(*lines.items())
    for asic in cal_df.keys():
        for ch in cal_df[asic].index:
            counts = histograms[asic][ch]
            gain, gain_err, offset = cal_df[asic].loc[ch][['gain', 'gain_err', 'offset']]
            try:
                guesses = [[lout_lim * PHT_KEV * lv * gain + offset for lout_lim in lout_guess] for lv in line_values]
                limits = estimate_peaks_from_guess(bins, counts, guess=guesses)
                centers, center_errs, *etc = fit_peaks(bins, counts, limits)
                light_outs, light_out_errs = compute_lout(centers, center_errs, gain, gain_err, offset, line_values)
            except DetectPeakError:
                flagged.setdefault(asic, []).append(ch)
            else:
                results_lo.setdefault(asic, {})[ch] = np.concatenate((light_outs, light_out_errs))
    return results_lo, flagged


def closest_peaks(guess, peaks, peaks_infos):
    peaks_dist_from_guess = [[dist_from_intv(peak, guess_lo, guess_hi) for peak in peaks]
                             for guess_lo, guess_hi in guess]
    best_peaks = peaks[np.argmin(peaks_dist_from_guess, axis=1)]
    return best_peaks, {key: val[np.isin(peaks, best_peaks)] for key, val in peaks_infos.items()}


def estimate_peaks_from_guess(bins, counts, guess):
    mm = move_mean(counts, 5)
    unfiltered_peaks, unfiltered_peaks_info = find_peaks(mm, prominence=10, width=5)
    if len(unfiltered_peaks) >= len(guess):
        peaks, peaks_info = closest_peaks(guess, unfiltered_peaks, unfiltered_peaks_info)
    else:
        raise DetectPeakError("candidate peaks are less than lines to fit.")
    limits = [(bins[int(p - w)], bins[int(p + w)]) for p, w in zip(peaks, peaks_info['widths'])]
    return np.array(limits).reshape(len(peaks), 2)


def compute_lout(centers, center_errs, gain, gain_err, offset, lines):
    light_outs = (centers - offset) / gain / PHT_KEV / lines
    light_out_errs = np.sqrt((center_errs / offset) ** 2
                             + (gain_err / offset) ** 2
                             + ((centers - gain) / offset) ** 2) / PHT_KEV / lines
    return light_outs, light_out_errs


def xcalibrate(bins, histograms, lines, onchannels):
    results_fit, results_cal, flagged = {}, {}, {}
    lines_keys, lines_values = zip(*lines.items())
    for asic in onchannels.keys():
        for ch in onchannels[asic]:
            counts = histograms[asic][ch]
            try:
                if len(lines_values) > 2:
                    limits = estimate_peaks_from_lratio(bins, counts, lines_values)
                else:
                    raise DetectPeakError("not enough lines to calibrate.")
                centers, center_errs, *etc = fit_peaks(bins, counts, limits)
                gain, gain_err, offset, offset_err, chi2 = calibrate_chn(centers, center_errs, lines_values)
            except DetectPeakError:
                flagged.setdefault(asic, []).append(ch)
            else:
                results_fit.setdefault(asic, {})[ch] = np.concatenate((centers, center_errs, *etc, *limits.T))
                results_cal.setdefault(asic, {})[ch] = np.array((gain, gain_err, offset, offset_err, chi2))
    return results_fit, results_cal, flagged


def filter_peaks_lratio(lines: list, peaks, peaks_infos):
    normalize = (lambda x: [(x[i + 1] - x[i]) / (x[-1] - x[0]) for i in range(len(x) - 1)])
    weight = (lambda x: [x[i + 1] * x[i] for i in range(len(x) - 1)])

    peaks_combinations = [*combinations(peaks, r=len(lines))]
    norm_ls = normalize(lines)
    norm_ps = [*map(normalize, peaks_combinations)]
    weights = [*map(weight, combinations(peaks_infos["prominences"], r=len(lines)))]
    loss = np.sum(np.square(np.array(norm_ps) - np.array(norm_ls)) / weights
                  , axis=1)
    best_peaks = peaks_combinations[np.argmin(loss)]
    return best_peaks, {key: val[np.isin(peaks, best_peaks)] for key, val in peaks_infos.items()}


def estimate_peaks_from_lratio(bins, counts, lines: list):
    mm = move_mean(counts, 5)
    unfiltered_peaks, unfiltered_peaks_info = find_peaks(mm, prominence=20, width=5)
    if len(unfiltered_peaks) >= len(lines):
        peaks, peaks_info = filter_peaks_lratio(lines, unfiltered_peaks, unfiltered_peaks_info)
    else:
        raise DetectPeakError("candidate peaks are less than lines to fit.")
    limits = [(bins[int(p - w)], bins[int(p + w)]) for p, w in zip(peaks, peaks_info['widths'])]
    return np.array(limits).reshape(len(peaks), 2)


def line_fitter(x, y, limits, bkg=None):
    """
    Fits a Gaussian line between start and stop
    Input:  x, y: input arrays
            limits = (start,stop): tuple with fit boundaries
            bkg (optional): polynomial background. If not None, should be polynomial degree (up to 7th deg.)
    Output: lmfit result object
    """
    start, stop = limits
    y_err = np.sqrt(y)
    x_start = np.where(x > start)[0][0]
    x_stop = np.where(x < stop)[0][-1]
    x_fit = x[x_start:x_stop]
    y_fit = y[x_start:x_stop]
    y_fit_err = y_err[x_start:x_stop]

    if bkg is not None:
        mod = GaussianModel() + PolynomialModel(bkg, prefix='bkg_')
    else:
        mod = GaussianModel()
    pars = mod.guess(y_fit, x=x_fit)
    center = x_fit[np.argmax(y_fit)]
    pars['center'].set(center, min=start, max=stop)
    result = mod.fit(y_fit, pars, x=x_fit, weights=y_fit_err)

    x_fine = np.linspace(x[0], x[-1], len(x) * 100)
    fitting_curve = mod.eval(x=x_fine,
                             amplitude=result.best_values['amplitude'],
                             center=result.best_values['center'],
                             sigma=result.best_values['sigma'])

    return result, start, stop, x_fine, fitting_curve


def fit_peaks(x, y, limits):
    """
    Fit the peaks in the given spectrum, each in the given limits

    Input:
            x, y : arrays with input spectrum
            limits : array of limits for each peak
    Output:
            fit_results : array, rows is each peak, column is mu, mu_err, fwhm, fwhm_err
            x_fine: array of adu. useful for visualization
            fitting_curve: array of fit values. useful for visualization
    """
    n_peaks = len(limits)
    centers = np.zeros(n_peaks)
    center_errs = np.zeros(n_peaks)
    fwhms = np.zeros(n_peaks)
    fwhm_errs = np.zeros(n_peaks)
    amps = np.zeros(n_peaks)
    amp_errs = np.zeros(n_peaks)

    for i in range(n_peaks):
        result, start, stop, x_fine, fitting_curve = line_fitter(x, y, limits[i])
        centers[i] = result.params['center'].value
        center_errs[i] = result.params['center'].stderr
        fwhms[i] = result.params['fwhm'].value
        fwhm_errs[i] = result.params['fwhm'].stderr
        amps[i] = result.params['amplitude'].value
        amp_errs[i] = result.params['amplitude'].stderr

    return centers, center_errs, fwhms, fwhm_errs, amps, amp_errs


def calibrate_chn(centers, center_errs, lines):
    """
    TODO: FUNCTION CHANGED. UPDATE DOCSTRING
    This function establish gain and offset values for each channel, with respective errors from the peaks fitted.

    Input:
          line_data : array containing the energy line and the color associated
          fit_results : array with the fit of the peaks
    Output:
          chi, gain, gain_err, offset, offset_err : floats
    """

    lmod = LinearModel()
    pars = lmod.guess(centers, x=lines)
    resultlin = lmod.fit(centers, pars, x=lines, weights=center_errs)

    chi2 = resultlin.redchi
    gain = resultlin.params['slope'].value
    offset = resultlin.params['intercept'].value
    gain_err = resultlin.params['slope'].stderr
    offset_err = resultlin.params['intercept'].stderr

    return gain, gain_err, offset, offset_err, chi2


# def histogram(data, start, nbins, step):
#     hists = {}
#     for asic in 'ABCD':
#         hist_asics = {}
#         quad_df = data[data['QUADID'] == asic]
#         for ch in range(32):
#             ch_data = quad_df[(quad_df['CHN'] == ch)]
#             counts, bins = np.histogram(ch_data['ADC'], range=(start, start + nbins * step), bins=nbins)
#             hist_asics[ch] = counts
#         hists[asic] = hist_asics
#     return bins, hists

def histogram(data, start, nbins, step, nthreads = 1):
    def helper(asic):
        hist_asics = {}
        quad_df = data[data['QUADID'] == asic]
        for ch in range(32):
            ch_data = quad_df[(quad_df['CHN'] == ch)]
            counts, bins = np.histogram(ch_data['ADC'], range=(start, start + nbins * step), bins=nbins)
            hist_asics[ch] = counts
        return asic, hist_asics, bins

    bins = np.linspace(start, start + nbins * step, nbins + 1)
    results = Parallel(n_jobs=nthreads)(delayed(helper)(asic) for asic in 'ABCD')
    hists = {key: value for key, value, _ in results}
    return bins, hists


def move_mean(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()


s2i = (lambda asic: "ABCD".find(str.upper(asic)))
i2s = (lambda n: chr(65 + n))


def add_evtype_tag(data, couples):
    """
    inplace add event type (X or S) column
    :param data:
    :return:
    """
    data['CHN'] = data['CHN'] + 1
    qm = data['QUADID'].map({key: 100 ** s2i(key) for key in 'ABCD'})
    chm_dict = dict(np.concatenate([(couples[key] + 1) * 100**s2i(key) for key in couples.keys()]))
    chm = data['CHN']*qm
    data.insert(loc=3, column='EVTYPE', value=(data
                                               .assign(CHN=chm.map(chm_dict).fillna(chm))
                                               .duplicated(['TIME', 'EVTID', 'CHN'], keep=False)
                                               .map({False: 'X', True: 'S'})
                                               .astype('string')))
    data['CHN'] = data['CHN'] - 1
    return data
