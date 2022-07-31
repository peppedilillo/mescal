from collections import namedtuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lmfit.models import GaussianModel
from source.errors import FailedFitError


PHT_KEV = 3.65 / 1000

histograms_collection = namedtuple('histogram', ['bins', 'counts'])


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
    errors = np.clip(np.sqrt(y_fit), 1, None)

    mod = GaussianModel()
    pars = mod.guess(y_fit, x=x_fit)
    try:
        result = mod.fit(y_fit, pars, x=x_fit, weights=1/errors)
    except TypeError:
        raise FailedFitError("peak fitter error.")
    x_fine = np.linspace(x[0], x[-1], len(x) * 100)
    fitting_curve = mod.eval(x=x_fine,
                             amplitude=result.best_values['amplitude'],
                             center=result.best_values['center'],
                             sigma=result.best_values['sigma'])

    return result, start, stop, x_fine, fitting_curve


def compute_histogram(data, start, end, nbins, nthreads=1):
    def helper(quad):
        hist_quads = {}
        quad_df = data[data['QUADID'] == quad]
        for ch in range(32):
            adcs = quad_df[(quad_df['CHN'] == ch)]['ADC']
            ys, _ = np.histogram(adcs, range=(start, end), bins=nbins)
            hist_quads[ch] = ys
        return quad, hist_quads

    bins = np.linspace(start, end, nbins + 1)
    results = Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in 'ABCD')
    counts = {key: value for key, value in results}
    return histograms_collection(bins, counts)


def move_mean(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()
