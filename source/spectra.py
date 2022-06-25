from itertools import combinations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lmfit.models import GaussianModel
from lmfit.models import LinearModel
from scipy.signal import find_peaks

PHT_KEV = 3.65 / 1000


class DetectPeakError(Exception):
    """An error while finding peaks."""


def make_calibrated_events_list(data, calibrated_sdds, calibrated_scintillators, scintillator_couples):
    columns = ['TIME', 'ENERGY', 'EVTYPE', 'CHN', 'QUADID']
    types = ['float64', 'float32', 'U1', 'int8', 'U1']
    dtypes = {col: tp for col, tp in zip(columns, types)}
    container = np.recarray(shape=0, dtype=[*dtypes.items()])

    disorganized_events = _get_calibrated_events(data, calibrated_sdds, calibrated_scintillators, scintillator_couples)
    for quadrant in disorganized_events.keys():
        x_events, gamma_events = disorganized_events[quadrant]
        xtimes, xenergies, xchannels, xquadrants, xevtypes = (*x_events.T,
                                                              np.array([quadrant] * len(x_events)),
                                                              np.array(['X'] * len(x_events)))
        stimes, senergies, schannels, squadrants, sevtypes = (*gamma_events.T,
                                                              np.array([quadrant] * len(gamma_events)),
                                                              np.array(['S'] * len(gamma_events)))
        x_array = np.rec.fromarrays([xtimes, xenergies, xevtypes, xchannels, xquadrants], dtype=[*dtypes.items()])
        s_array = np.rec.fromarrays([stimes, senergies, sevtypes, schannels, squadrants], dtype=[*dtypes.items()])
        container = np.hstack((container, x_array, s_array))
    out = pd.DataFrame(container).sort_values('TIME').reset_index(drop=True)
    return out


def _extract_gamma_events(quadrant_data, calibrated_scintillators, scintillator_couples):
    scintillator_events = quadrant_data[quadrant_data['EVTYPE'] == 'S']
    scintillator_events = scintillator_events.assign(CHN=scintillator_events['CHN']
                                                     .map(dict(scintillator_couples))
                                                     .fillna(scintillator_events['CHN']))
    simultaneous_scintillator_events = scintillator_events.groupby(['TIME', 'CHN'])
    times, channels = np.array([*simultaneous_scintillator_events.groups.keys()]).T
    companion_channels = pd.Series(channels) \
        .map({v: k for k, v in dict(scintillator_couples).items()}).values
    xenergy_sum = simultaneous_scintillator_events.sum()['XENS'].values
    scintillator_light_outputs = calibrated_scintillators['light_out'].loc[channels].values + \
                                 calibrated_scintillators['light_out'].loc[companion_channels].values

    gamma_energies = xenergy_sum / scintillator_light_outputs / PHT_KEV
    return np.column_stack((times, gamma_energies, channels))


def _extract_x_events(quadrant_data):
    return quadrant_data[quadrant_data['EVTYPE'] == 'X'][['TIME', 'XENS', 'CHN']].values


def _insert_xenergy_column(data, calibrated_sdds):
    adcs = data['ADC']
    chns = data['CHN']
    offsets = calibrated_sdds.loc[chns]['offset'].values
    gains = calibrated_sdds.loc[chns]['gain'].values

    xenergies = (adcs - offsets) / gains
    data.insert(0, 'XENS', xenergies)
    return data


def _get_calibrated_events(data, calibrated_sdds, calibrated_scintillators, scintillator_couples):
    out = {}
    for quadrant in calibrated_scintillators.keys():
        quadrant_data = data[(data['QUADID'] == quadrant) &
                             (data['CHN'].isin(calibrated_scintillators[quadrant].index))]
        quadrant_data = _insert_xenergy_column(quadrant_data, calibrated_sdds[quadrant])

        x_events = _extract_x_events(quadrant_data)
        gamma_events = _extract_gamma_events(quadrant_data,
                                             calibrated_scintillators[quadrant],
                                             scintillator_couples[quadrant])

        out[quadrant] = (x_events, gamma_events)
    return out


def scalibrate(bins, histograms, cal_df, lines, lout_guess):
    results_fit, results_slo, flagged = {}, {}, {}
    line_keys, line_values = zip(*lines.items())
    for asic in cal_df.keys():
        for ch in cal_df[asic].index:
            counts = histograms[asic][ch]
            gain, gain_err, offset = cal_df[asic].loc[ch][['gain', 'gain_err', 'offset']]
            try:
                guesses = [[lout_lim * PHT_KEV * lv * gain + offset for lout_lim in lout_guess] for lv in line_values]
                limits = _estimate_peaks_from_guess(bins, counts, guess=guesses)
                centers, center_errs, *etc = _fit_peaks(bins, counts, limits)
                los, lo_errs = _compute_louts(centers, center_errs, gain, gain_err, offset, line_values)
                lo, lo_err = _do_something_to_deal_with_the_fact_that_you_may_have_many_gamma_lines(los, lo_errs)
            except DetectPeakError:
                flagged.setdefault(asic, []).append(ch)
            else:
                results_fit.setdefault(asic, {})[ch] = np.column_stack((centers, center_errs, *etc, *limits.T)).flatten()
                results_slo.setdefault(asic, {})[ch] = np.array((lo, lo_err))
    return results_fit, results_slo, flagged


def _do_something_to_deal_with_the_fact_that_you_may_have_many_gamma_lines(light_outs, light_outs_errs):
    # in doubt mean
    return light_outs.mean(), np.sqrt(np.sum(light_outs_errs**2))


_dist_from_intv = (lambda x, lo, hi: abs((x - lo) + (x - hi)))


def _closest_peaks(guess, peaks, peaks_infos):
    peaks_dist_from_guess = [[_dist_from_intv(peak, guess_lo, guess_hi) for peak in peaks]
                             for guess_lo, guess_hi in guess]
    argmin = np.argmin(peaks_dist_from_guess, axis=1)
    best_peaks = peaks[argmin]
    return best_peaks, {key: val[argmin] for key, val in peaks_infos.items()}


def _estimate_peaks_from_guess(bins, counts, guess, limratio=(1, 1)):
    mm = move_mean(counts, 5)
    unfiltered_peaks, unfiltered_peaks_info = find_peaks(mm, prominence=10, width=5)
    if len(unfiltered_peaks) >= len(guess):
        peaks, peaks_info = _closest_peaks(guess, unfiltered_peaks, unfiltered_peaks_info)
    else:
        raise DetectPeakError("candidate peaks are less than lines to fit.")
    lo_r, hi_r = limratio
    limits = [(bins[int(p - w*lo_r)], bins[int(p + w*hi_r)]) for p, w in zip(peaks, peaks_info['widths'])]
    return np.array(limits).reshape(len(peaks), 2)


def _compute_louts(centers, center_errs, gain, gain_err, offset, lines):
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
                    limits = _estimate_peakpos_from_lratio(bins, counts, lines_values)
                else:
                    raise DetectPeakError("not enough lines to calibrate.")
                centers, center_errs, *etc = _fit_peaks(bins, counts, limits, weights='amplitude')
                gain, gain_err, offset, offset_err, chi2 = _calibrate_chn(centers, center_errs, lines_values)
            except DetectPeakError:
                flagged.setdefault(asic, []).append(ch)
            else:
                results_fit.setdefault(asic, {})[ch] = np.column_stack((centers, center_errs, *etc, *limits.T)).flatten()
                results_cal.setdefault(asic, {})[ch] = np.array((gain, gain_err, offset, offset_err, chi2))
    return results_fit, results_cal, flagged


def _filter_peaks_lratio(lines: list, peaks, peaks_infos):
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


def _estimate_peakpos_from_lratio(bins, counts, lines: list):
    mm = move_mean(counts, 5)
    unfiltered_peaks, unfiltered_peaks_info = find_peaks(mm, prominence=20, width=5)
    if len(unfiltered_peaks) >= len(lines):
        peaks, peaks_info = _filter_peaks_lratio(lines, unfiltered_peaks, unfiltered_peaks_info)
    else:
        raise DetectPeakError("candidate peaks are less than lines to fit.")
    limits = [(bins[int(p - w)], bins[int(p + w)]) for p, w in zip(peaks, peaks_info['widths'])]
    return np.array(limits).reshape(len(peaks), 2)


def _line_fitter(x, y, limits):
    start, stop = limits
    x_start = np.where(x > start)[0][0]
    x_stop = np.where(x < stop)[0][-1]
    x_fit = (x[x_start:x_stop + 1][1:] + x[x_start:x_stop + 1][:-1])/2
    y_fit = y[x_start:x_stop]
    weights = np.sqrt(y_fit)

    mod = GaussianModel()
    pars = mod.guess(y_fit, x=x_fit)

    result = mod.fit(y_fit, pars, x=x_fit, weights=weights)
    x_fine = np.linspace(x[0], x[-1], len(x) * 100)
    fitting_curve = mod.eval(x=x_fine,
                             amplitude=result.best_values['amplitude'],
                             center=result.best_values['center'],
                             sigma=result.best_values['sigma'])

    return result, start, stop, x_fine, fitting_curve


def _fit_peaks(x, y, limits, weights=None):
    n_peaks = len(limits)
    centers = np.zeros(n_peaks)
    center_errs = np.zeros(n_peaks)
    fwhms = np.zeros(n_peaks)
    fwhm_errs = np.zeros(n_peaks)
    amps = np.zeros(n_peaks)
    amp_errs = np.zeros(n_peaks)

    for i in range(n_peaks):
        result, start, stop, x_fine, fitting_curve = _line_fitter(x, y, limits[i])
        centers[i] = result.params['center'].value
        center_errs[i] = result.params['center'].stderr
        fwhms[i] = result.params['fwhm'].value
        fwhm_errs[i] = result.params['fwhm'].stderr
        amps[i] = result.params['amplitude'].value
        amp_errs[i] = result.params['amplitude'].stderr

    return centers, center_errs, fwhms, fwhm_errs, amps, amp_errs


def _calibrate_chn(centers, center_errs, lines):
    lmod = LinearModel()
    pars = lmod.guess(centers, x=lines)
    resultlin = lmod.fit(centers, pars, x=lines, weights=center_errs)

    chi2 = resultlin.redchi
    gain = resultlin.params['slope'].value
    offset = resultlin.params['intercept'].value
    gain_err = resultlin.params['slope'].stderr
    offset_err = resultlin.params['intercept'].stderr

    return gain, gain_err, offset, offset_err, chi2


def histogram(data, start, nbins, step, nthreads = 1):
    def helper(quad):
        hist_quads = {}
        quad_df = data[data['QUADID'] == quad]
        for ch in range(32):
            ch_data = quad_df[(quad_df['CHN'] == ch)]
            counts, bins = np.histogram(ch_data['ADC'], range=(start, start + nbins * step), bins=nbins)
            hist_quads[ch] = counts
        return quad, hist_quads, bins

    bins = np.linspace(start, start + nbins * step, nbins + 1)
    results = Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in 'ABCD')
    hists = {key: value for key, value, _ in results}
    return bins, hists


def move_mean(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()


s2i = (lambda quad: "ABCD".find(str.upper(quad)))
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
                                               .duplicated(['SID', 'CHN'], keep=False)
                                               .map({False: 'X', True: 'S'})
                                               .astype('string')))
    data['CHN'] = data['CHN'] - 1
    return data