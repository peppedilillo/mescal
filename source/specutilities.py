import numpy as np
import pandas as pd
from lmfit.models import GaussianModel, LinearModel, PolynomialModel
from scipy.signal import find_peaks


def move_mean(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()


def filter_peaks(lines, peaks, peaks_infos):
    lines_ratio = np.array([(l2 - l1) / (l3 - l2) for l1, l2, l3 in zip(lines[:-2], lines[1:-1], lines[2:])])
    peaks_ratio = np.array([(l2 - l1) / (l3 - l2) for l1, l2, l3 in zip(peaks[:-2], peaks[1:-1], peaks[2:])])
    n = np.argmin(np.abs(peaks_ratio - lines_ratio))
    return peaks[n:n + 3], {key: val[n:n + 3] for key, val in peaks_infos.items()}


def detect_peaks(bins, counts, lines):
    mm = move_mean(move_mean(counts, 10), 1)
    unfiltered_peak, unfiltered_peaks_info = find_peaks(mm, prominence=50, width=5)
    if len(unfiltered_peak) > 2:
        peaks, peaks_info = filter_peaks(lines, unfiltered_peak, unfiltered_peaks_info)
    else:
        raise ValueError("Will get there.")
    limits = [(bins[int(p - w)], bins[int(p + w)]) for p, w in zip(peaks, peaks_info['widths'])]
    return limits


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
    fitting_curve = mod.eval(x=x_fine, \
                             amplitude=result.best_values['amplitude'], \
                             center=result.best_values['center'], \
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


def calibrate(centers, center_errs, lines):
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
