from collections import namedtuple
from itertools import combinations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lmfit.models import GaussianModel
from lmfit.models import LinearModel
from scipy.signal import find_peaks
from source.errors import DetectPeakError
from source.errors import FailedFitError
from source.errors import warn_failed_peak_detection
from source.errors import warn_failed_linearity_fit
from source.errors import warn_failed_peak_fit
import logging

PHT_KEV = 3.65 / 1000

histograms_collection = namedtuple('histogram', ['bins', 'counts'])


def make_events_list(data, calibrated_sdds, calibrated_scintillators, scintillator_couples, nthreads=1,):
    columns = ['TIME', 'ENERGY', 'EVTYPE', 'CHN', 'QUADID']
    types = ['float64', 'float32', 'U1', 'int8', 'U1']
    dtypes = {col: tp for col, tp in zip(columns, types)}
    container = np.recarray(shape=0, dtype=[*dtypes.items()])

    disorganized_events = _get_calibrated_events(data,
                                                 calibrated_sdds,
                                                 calibrated_scintillators,
                                                 scintillator_couples,
                                                 nthreads=nthreads)

    for quadrant in disorganized_events.keys():
        x_events, gamma_events = disorganized_events[quadrant]

        xtimes, xenergies, xchannels = x_events.T
        xquadrants = np.array([quadrant] * len(x_events))
        xevtypes = np.array(['X'] * len(x_events))

        stimes, senergies, schannels = gamma_events.T
        squadrants = np.array([quadrant] * len(gamma_events))
        sevtypes = np.array(['S'] * len(gamma_events))

        x_array = np.rec.fromarrays(
            [xtimes, xenergies, xevtypes, xchannels, xquadrants],
            dtype=[*dtypes.items()]
        )
        s_array = np.rec.fromarrays(
            [stimes, senergies, sevtypes, schannels, squadrants],
            dtype=[*dtypes.items()]
        )
        container = np.hstack((container, x_array, s_array))
    out = pd.DataFrame(container).sort_values('TIME').reset_index(drop=True)
    return out


def _get_coupled_channels(channels, couples):
    """
    Returns a subset of channels filtered of the "widow" channels

    Args:
        channels: a list of channels
        couples: a list of channels couples (e.g., two elements lists or tuples)

    Returns:
        a list  of channels.
    """
    coupled_channels = []
    for couple in couples:
        if all([channel in channels for channel in couple]):
            coupled_channels += [ch for ch in couple]
    return coupled_channels


def _get_calibrated_events(data, calibrated_sdds, calibrated_scintillators, scintillator_couples, nthreads=1):
    def helper(quadrant):
        couples = scintillator_couples[quadrant]
        calibrated_channels = calibrated_scintillators[quadrant].index
        coupled_channels = _get_coupled_channels(calibrated_channels, couples)

        quadrant_data = data[(data['QUADID'] == quadrant) &
                             (data['CHN'].isin(coupled_channels))]
        quadrant_data = _insert_xenergy_column(quadrant_data, calibrated_sdds[quadrant])

        x_events = _extract_x_events(quadrant_data)
        gamma_events = _extract_gamma_events(quadrant_data,
                                             calibrated_scintillators[quadrant],
                                             scintillator_couples[quadrant])

        return quadrant, (x_events, gamma_events)

    results = Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in calibrated_scintillators.keys())
    return {quadrant: value for quadrant, value in results}


def _extract_gamma_events(quadrant_data, calibrated_scintillators, scintillator_couples):
    gamma_events = quadrant_data[quadrant_data['EVTYPE'] == 'S']

    channels = gamma_events['CHN']
    companion_to_chn = dict(scintillator_couples)
    same_value_if_coupled = gamma_events['CHN'].map(companion_to_chn).fillna(channels)
    gamma_events = gamma_events.assign(CHN=same_value_if_coupled)

    simultaneous_scintillator_events = gamma_events.groupby(['TIME', 'CHN'])
    times, channels = np.array([*simultaneous_scintillator_events.groups.keys()]).T

    channel_to_companion = {v: k for k, v in dict(scintillator_couples).items()}
    companion_channels = pd.Series(channels).map(channel_to_companion).values
    xenergy_sum = simultaneous_scintillator_events.sum()['XENS'].values
    channels_lo = calibrated_scintillators['light_out'].loc[channels].values
    companions_lo = calibrated_scintillators['light_out'].loc[companion_channels].values

    light_output_sum = channels_lo + companions_lo
    gamma_energies = xenergy_sum / light_output_sum / PHT_KEV
    calibrated_gamma_events = np.column_stack((times, gamma_energies, channels))
    return calibrated_gamma_events


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


def scalibrate(histograms, cal_df, radsources, lout_guess):
    results_fit, results_slo, flagged = {}, {}, {}
    radsources_energies = [l.energy for l in radsources.values()]

    bins = histograms.bins
    for quad in cal_df.keys():
        for ch in cal_df[quad].index:
            counts = histograms.counts[quad][ch]
            gain = cal_df[quad].loc[ch]['gain']
            gain_err = cal_df[quad].loc[ch]['gain_err']
            offset = cal_df[quad].loc[ch]['offset']
            offset_err = cal_df[quad].loc[ch]['offset_err']

            guesses = [[lout_lim * PHT_KEV * lv * gain + offset
                        for lout_lim in lout_guess]
                       for lv in radsources_energies]

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
                intervals, (centers, center_errs, *etc) = _fit_radsources_peaks(
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

            los, lo_errs = _compute_louts(
                centers,
                center_errs,
                gain,
                gain_err,
                offset,
                offset_err,
                radsources_energies,
            )
            lo, lo_err = deal_with_multiple_gamma_decays(los, lo_errs)

            int_inf, int_sup = zip(*intervals)
            results_fit.setdefault(quad, {})[ch] = np.column_stack(
                (centers, center_errs, *etc, int_inf, int_sup)).flatten()
            results_slo.setdefault(quad, {})[ch] = np.array((lo, lo_err))
    return results_fit, results_slo, flagged


def deal_with_multiple_gamma_decays(light_outs, light_outs_errs):
    # in doubt mean
    return light_outs.mean(), np.sqrt(np.sum(light_outs_errs ** 2))


def _dist_from_intv(x, lo, hi): return abs((x - lo) + (x - hi))


def _closest_peaks(guess, peaks, peaks_infos):
    peaks_dist_from_guess = [[_dist_from_intv(peak, guess_lo, guess_hi)
                              for peak in peaks]
                             for guess_lo, guess_hi in guess]
    argmin = np.argmin(peaks_dist_from_guess, axis=1)
    best_peaks = peaks[argmin]
    best_peaks_infos = {key: val[argmin] for key, val in peaks_infos.items()}
    return best_peaks, best_peaks_infos


def _estimate_peaks_from_guess(bins, counts, guess):
    window_len = 20
    prominence = 5
    width = 20

    mm = move_mean(counts, window_len)
    many_peaks, many_peaks_info = find_peaks(mm, prominence=prominence, width=width)
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


def xcalibrate(histograms, radsources, channels, default_calibration=None):
    results_xfit, results_cal, flagged = {}, {}, {}
    radsources_energies = [l.energy for l in radsources.values()]

    for quad in channels.keys():
        for ch in channels[quad]:
            bins = histograms.bins
            counts = histograms.counts[quad][ch]

            def packaged_calib(): return default_calibration[quad].loc[ch]
            try:
                limits = _find_peaks_limits(
                    bins,
                    counts,
                    radsources_energies,
                    packaged_calib,
                )
            except DetectPeakError:
                message = warn_failed_peak_detection(quad, ch)
                logging.warning(message)
                flagged.setdefault(quad, []).append(ch)
                continue

            try:
                intervals, fit_results = _fit_radsources_peaks(
                    bins,
                    counts,
                    limits,
                    radsources,
                )
            except FailedFitError:
                meassage = warn_failed_peak_fit(quad,ch)
                logging.warning(meassage)
                flagged.setdefault(quad, []).append(ch)
                continue

            centers, center_errs, *etc = fit_results
            int_inf, int_sup = zip(*intervals)

            try:
                cal_results = _calibrate_chn(
                    centers,
                    radsources_energies,
                    weights=center_errs,
                )
            except FailedFitError:
                message = warn_failed_linearity_fit(quad, ch)
                logging.warning(message)
                flagged.setdefault(quad, []).append(ch)
                continue

            results_xfit.setdefault(quad, {})[ch] = np.column_stack(
                (*fit_results, int_inf, int_sup)).flatten()
            results_cal.setdefault(quad, {})[ch] = np.array(cal_results)
    return results_xfit, results_cal, flagged


def _find_peaks_limits(bins, counts, radsources: list, unpack_calibration):
    try:
        channel_calib = unpack_calibration()
    except TypeError:
        return _lims_from_decays_ratio(bins, counts, radsources)
    else:
        return _lims_from_existing_calib(bins, counts, radsources, channel_calib)


def _lims_from_existing_calib(bins, counts, radsources: list, channel_calib):
    window_len = 5
    width = 5
    prominence = 5
    distance = 5
    low_en_thr = 1.0  # keV

    energies = (bins - channel_calib['offset'])/ channel_calib['gain']
    (inf_bin, *_), = np.where(energies > low_en_thr)
    smoothed_counts = move_mean(counts, window_len)
    unfiltered_peaks, unfiltered_peaks_info = find_peaks(
        smoothed_counts,
        prominence=prominence,
        width=width,
        distance=distance,
    )
    enfiltered_peaks, enfiltered_peaks_info = _filter_peaks_low_energy(
        inf_bin,
        unfiltered_peaks,
        unfiltered_peaks_info,
    )
    if len(enfiltered_peaks) < len(radsources):
        raise DetectPeakError("candidate peaks are less than radsources to fit.")
    peaks, peaks_info = _filter_peaks_proximity(
        radsources,
        energies,
        enfiltered_peaks,
        enfiltered_peaks_info,
    )
    limits = [(bins[int(p - w)], bins[int(p + w)])
              for p, w in zip(peaks, peaks_info['widths'])]
    return limits


def _filter_peaks_proximity(radsources: list, energies, peaks, peaks_infos):
    peaks_combinations = [*combinations(peaks, r=len(radsources))]
    enpeaks_combinations = np.take(energies, peaks_combinations)
    loss = np.sum(np.square(enpeaks_combinations - np.array(radsources)), axis=1)
    filtered_peaks = peaks_combinations[np.argmin(loss)]
    filtered_peaks_info = {key: val[np.isin(peaks, filtered_peaks)]
                           for key, val in peaks_infos.items()}
    return filtered_peaks, filtered_peaks_info


def _filter_peaks_low_energy(lim_bin, peaks, peaks_infos):
    filtered_peaks = peaks[np.where(peaks > lim_bin)]
    filtered_peaks_info = {key: val[np.isin(peaks, filtered_peaks)]
                           for key, val in peaks_infos.items()}
    return filtered_peaks, filtered_peaks_info


def _lims_from_decays_ratio(bins, counts, radsources: list):
    window_len = 5
    prominence = 10
    width = 5
    distance = 10

    if len(radsources) < 3:
        raise DetectPeakError("not enough radsources to calibrate.")
    mm = move_mean(counts, window_len)
    unfiltered_peaks, unfiltered_peaks_info = find_peaks(
        mm,
        prominence=prominence,
        width=width,
        distance=distance,
    )
    if len(unfiltered_peaks) < len(radsources):
        raise DetectPeakError("candidate peaks are less than radsources to fit.")
    peaks, peaks_info = _filter_peaks_lratio(radsources, unfiltered_peaks, unfiltered_peaks_info)
    limits = [(bins[int(p - w)], bins[int(p + w)])
              for p, w in zip(peaks, peaks_info['widths'])]
    return limits


def _filter_peaks_lratio(radsources: list, peaks, peaks_infos):
    normalize = (lambda x: [(x[i + 1] - x[i]) / (x[-1] - x[0]) for i in range(len(x) - 1)])
    weight = (lambda x: [x[i + 1] * x[i] for i in range(len(x) - 1)])

    peaks_combinations = [*combinations(peaks, r=len(radsources))]
    proms_combinations = combinations(peaks_infos["prominences"], r=len(radsources))
    norm_ls = normalize(radsources)
    norm_ps = [*map(normalize, peaks_combinations)]
    weights = [*map(weight, proms_combinations)]
    loss = np.sum(np.square(np.array(norm_ps) - np.array(norm_ls))
                  #/ np.square(weights)
                  , axis=1)
    best_peaks = peaks_combinations[np.argmin(loss)]
    best_peaks_info = {key: val[np.isin(peaks, best_peaks)] for key, val in peaks_infos.items()}
    return best_peaks, best_peaks_info


def _fit_radsources_peaks(x, y, limits, radsources):
    centers, _, fwhms, _, *_ = _fit_peaks(x, y, limits)
    sigmas = fwhms/2.35
    lower, upper = zip(*[(rs.low_lim, rs.hi_lim) for rs in radsources.values()])
    intervals = [*zip(centers + sigmas*lower, centers + sigmas*upper)]
    fit_results = _fit_peaks(x, y, intervals)
    return intervals, fit_results


def _fit_peaks(x, y, limits):
    n_peaks = len(limits)
    centers = np.zeros(n_peaks)
    center_errs = np.zeros(n_peaks)
    fwhms = np.zeros(n_peaks)
    fwhm_errs = np.zeros(n_peaks)
    amps = np.zeros(n_peaks)
    amp_errs = np.zeros(n_peaks)

    for i in range(n_peaks):
        low_lim, hi_lim = limits[i]
        result, start, stop, x_fine, fitting_curve = _peak_fitter(x, y, (low_lim, hi_lim))
        centers[i] = result.params['center'].value
        center_errs[i] = result.params['center'].stderr
        fwhms[i] = result.params['fwhm'].value
        fwhm_errs[i] = result.params['fwhm'].stderr
        amps[i] = result.params['amplitude'].value
        amp_errs[i] = result.params['amplitude'].stderr

        if not (low_lim < centers[i] < hi_lim):
            raise FailedFitError("peaks limits do not contain fit center")
    return centers, center_errs, fwhms, fwhm_errs, amps, amp_errs


def _peak_fitter(x, y, limits):
    start, stop = limits
    x_start = np.where(x >= start)[0][0]
    x_stop = np.where(x < stop)[0][-1]
    x_fit = (x[x_start:x_stop + 1][1:] + x[x_start:x_stop + 1][:-1]) / 2
    y_fit = y[x_start:x_stop]
    weights = np.sqrt(y_fit)

    mod = GaussianModel()
    pars = mod.guess(y_fit, x=x_fit)

    try:
        result = mod.fit(y_fit, pars, x=x_fit, weights=weights)
    except TypeError:
        raise FailedFitError("peak fitter error.")
    x_fine = np.linspace(x[0], x[-1], len(x) * 100)
    fitting_curve = mod.eval(x=x_fine,
                             amplitude=result.best_values['amplitude'],
                             center=result.best_values['center'],
                             sigma=result.best_values['sigma'])

    return result, start, stop, x_fine, fitting_curve


def _calibrate_chn(centers, radsources: list, weights=None):
    lmod = LinearModel()
    pars = lmod.guess(centers, x=radsources)
    try:
        resultlin = lmod.fit(centers, pars, x=radsources, weights=weights)
    except ValueError:
        raise FailedFitError("linear fitter error")

    chi2 = resultlin.redchi
    gain = resultlin.params['slope'].value
    offset = resultlin.params['intercept'].value
    gain_err = resultlin.params['slope'].stderr
    offset_err = resultlin.params['intercept'].stderr

    return gain, gain_err, offset, offset_err, chi2


def compute_histogram(data, start, end, nbins, nthreads=1):
    def helper(quad):
        hist_quads = {}
        quad_df = data[data['QUADID'] == quad]
        for ch in range(32):
            adcs = quad_df[(quad_df['CHN'] == ch)]['ADC']
            counts, bins = np.histogram(adcs, range=(start, end), bins=nbins)
            hist_quads[ch] = counts
        return quad, hist_quads, bins

    bins = np.linspace(start, end, nbins + 1)
    results = Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in 'ABCD')
    counts = {key: value for key, value, _ in results}
    return histograms_collection(bins, counts)


def move_mean(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()

