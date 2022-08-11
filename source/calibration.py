import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lmfit.models import GaussianModel, LinearModel

import source.errors as err
from source.constants import PHOTOEL_PER_KEV
from source.eventlist import electrons_to_energy, make_electron_list
from source.inventory import fetch_default_sdd_calibration
from source.speaks import _estimate_peaks_from_guess
from source.xpeaks import find_xlimits

FIT_PARAMS = [
    "center",
    "center_err",
    "fwhm",
    "fwhm_err",
    "amp",
    "amp_err",
    "lim_low",
    "lim_high",
]

CAL_PARAMS = [
    "gain",
    "gain_err",
    "offset",
    "offset_err",
    "chi2",
]

LO_PARAMS = [
    "light_out",
    "light_out_err",
]

histogram = namedtuple("histogram", ["bins", "counts"])


def as_fit_dataframe(f):
    def wrapper(*args):
        nested_dict, radsources = f(*args)
        quadrants = nested_dict.keys()
        index = pd.MultiIndex.from_product(
            (radsources.keys(), FIT_PARAMS),
            names=["source", "parameter"],
        )

        dict_of_dfs = {
            q: pd.DataFrame(
                nested_dict[q],
                index=index,
            ).T.rename_axis("channel")
            for q in quadrants
        }
        return dict_of_dfs

    return wrapper


def as_cal_dataframe(f):
    def wrapper(*args):
        nested_dict = f(*args)
        quadrants = nested_dict.keys()

        dict_of_dfs = {
            q: pd.DataFrame(
                nested_dict[q],
                index=CAL_PARAMS,
            ).T.rename_axis("channel")
            for q in quadrants
        }
        return dict_of_dfs

    return wrapper


def as_slo_dataframe(f):
    def wrapper(*args):
        nested_dict = f(*args)
        quadrants = nested_dict.keys()

        dict_of_dfs = {
            q: pd.DataFrame(
                nested_dict[q],
                index=LO_PARAMS,
            ).T.rename_axis("channel")
            for q in quadrants
        }
        return dict_of_dfs

    return wrapper


def linrange(start, stop, step):
    nbins = int((stop - start) / step)
    end = start + nbins * step
    bins = np.linspace(start, end, nbins + 1)
    return bins


class Calibration:
    adc_bins = linrange(15000, 28000, 10)
    electron_bins = linrange(1000, 25000, 50)
    light_out_guess = (20.0, 30.0)

    def __init__(
        self,
        channels,
        couples,
        radsources,
        detector_model=None,
        temperature=None,
        console=None,
        nthreads=1,
    ):
        self.radsources = radsources
        self.channels = channels
        self.couples = couples
        self.detector_model = detector_model
        self.temperature = temperature
        self.xhistograms = None
        self.shistograms = None
        self.ehistograms = None
        self.xfit = {}
        self.sfit = {}
        self.efit = {}
        self.sdd_cal = {}
        self.scint_cal = {}
        self.optical_coupling = {}
        self.flagged = {}
        self.console = console
        self.nthreads = nthreads

    def __call__(self, data):
        self.xhistograms = self._make_xhistograms(data)
        self.shistograms = self._make_shistograms(data)
        self._print(":white_check_mark: Binned data.")

        if not self._xradsources():
            return
        self.xfit = self._fit_xradsources()
        self.sdd_cal = self._calibrate_sdds()
        self._print(":white_check_mark: Analyzed X events.")

        if not self._sradsources():
            return
        self.sfit = self._fit_sradsources()
        electron_evlist = make_electron_list(
            data,
            self.sdd_cal,
            self.sfit,
            self.couples,
            self.nthreads,
        )
        self.ehistograms = self._make_ehistograms(electron_evlist)
        self.efit = self._fit_gamma_electrons()
        self.scint_cal = self._calibrate_scintillators()
        self.optical_coupling = self._compute_effective_light_outputs()
        self._print(":white_check_mark: Analyzed gamma events.")

        if not self.scint_cal:
            return
        eventlist = electrons_to_energy(electron_evlist, self.scint_cal, self.couples)

        return eventlist

    def _print(self, message):
        if self.console:
            self.console.log(message)
        else:
            print(message)

    def _xradsources(self):
        xradsources, _ = self.radsources
        return xradsources

    def _sradsources(self):
        _, sradsources = self.radsources
        return sradsources

    def _flag(self, quad, chn, flag):
        self.flagged.setdefault(flag, []).append((quad, chn))

    def _fetch_hint(self):
        hint, key = fetch_default_sdd_calibration(
            self.detector_model,
            self.temperature,
        )
        self._print(":open_book: Loaded detection hints for {}@{}°C.".format(*key))
        return hint

    def _make_xhistograms(self, data):
        value = "ADC"
        bins = self.adc_bins
        data = data[data["EVTYPE"] == "X"]
        histograms = compute_histogram(
            value,
            data,
            bins,
            nthreads=self.nthreads,
        )
        return histograms

    def _make_shistograms(self, data):
        value = "ADC"
        bins = self.adc_bins
        data = data[data["EVTYPE"] == "S"]
        histograms = compute_histogram(
            value,
            data,
            bins,
            nthreads=self.nthreads,
        )
        return histograms

    def _make_ehistograms(self, electron_evlist):
        value = "ELECTRONS"
        data = electron_evlist[electron_evlist["EVTYPE"] == "S"]
        bins = self.electron_bins
        histograms = compute_histogram(
            value,
            data,
            bins,
            nthreads=self.nthreads,
        )
        return histograms

    @as_fit_dataframe
    def _fit_xradsources(self):
        bins = self.xhistograms.bins
        radiation_sources = self._xradsources()
        energies = [s.energy for s in radiation_sources.values()]
        constrains = [(s.low_lim, s.hi_lim) for s in radiation_sources.values()]
        # TODO still don't like how we deal with hints
        if self.detector_model:
            hint = self._fetch_hint()
        else:
            hint = None

        results = {}
        for quad in self.channels.keys():
            for ch in self.channels[quad]:
                counts = self.xhistograms.counts[quad][ch]

                if hint:
                    try:
                        gain_guess, offset_guess = hint[quad].loc[ch][
                            ["gain", "offset"]
                        ]
                    except KeyError:
                        message = err.warn_missing_defcal(quad, ch)
                        logging.warning(message)
                        self._flag(quad, ch, "defcal")
                        continue
                else:
                    gain_guess, offset_guess = None, None

                try:
                    limits = find_xlimits(
                        bins,
                        counts,
                        energies,
                        gain_guess,
                        offset_guess,
                    )
                except err.DetectPeakError:
                    message = err.warn_failed_peak_detection(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "xpeak")
                    continue
                except err.DefaultCalibNotFoundError as e:
                    message = err.warn_missing_defcal(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "xpeak")
                    continue

                try:
                    intervals, fit_results = self._fit_radsources_peaks(
                        bins,
                        counts,
                        limits,
                        constrains,
                    )
                except err.FailedFitError:
                    meassage = err.warn_failed_peak_fit(quad, ch)
                    logging.warning(meassage)
                    self._flag(quad, ch, "xfit")
                    continue

                int_inf, int_sup = zip(*intervals)
                results.setdefault(quad, {})[ch] = np.column_stack(
                    (*fit_results, int_inf, int_sup)
                ).flatten()
        return results, radiation_sources

    @as_fit_dataframe
    def _fit_sradsources(self):
        bins = self.shistograms.bins
        radiation_sources = self._sradsources()
        energies = [s.energy for s in radiation_sources.values()]
        constrains = [(s.low_lim, s.hi_lim) for s in radiation_sources.values()]

        results = {}
        for quad in self.sdd_cal.keys():
            for ch in self.sdd_cal[quad].index:
                counts = self.shistograms.counts[quad][ch]
                gain = self.sdd_cal[quad].loc[ch]["gain"]
                offset = self.sdd_cal[quad].loc[ch]["offset"]

                guesses = [
                    [
                        (0.5 * lout_lim) * PHOTOEL_PER_KEV * lv * gain + offset
                        for lout_lim in self.light_out_guess
                    ]
                    for lv in energies
                ]

                try:
                    limits = _estimate_peaks_from_guess(
                        bins,
                        counts,
                        guess=guesses,
                    )
                except err.DetectPeakError:
                    message = err.warn_failed_peak_detection(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "speak")
                    continue

                try:
                    intervals, fit_results = self._fit_radsources_peaks(
                        bins,
                        counts,
                        limits,
                        constrains,
                    )
                except err.FailedFitError:
                    message = err.warn_failed_peak_fit(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "sfit")
                    continue

                int_inf, int_sup = zip(*intervals)
                results.setdefault(quad, {})[ch] = np.column_stack(
                    (*fit_results, int_inf, int_sup)
                ).flatten()
        return results, radiation_sources

    @as_fit_dataframe
    def _fit_gamma_electrons(self):
        bins = self.ehistograms.bins
        radiation_sources = self._sradsources()
        energies = [s.energy for s in radiation_sources.values()]
        constrains = [(s.low_lim, s.hi_lim) for s in radiation_sources.values()]
        guesses = [
            [lout_lim * lv for lout_lim in self.light_out_guess] for lv in energies
        ]

        results = {}
        for quad in self.sfit.keys():
            for ch in self.sfit[quad].index:
                if ch not in self.couples[quad].keys():
                    continue
                scint = ch

                counts = self.ehistograms.counts[quad][scint]
                try:
                    limits = _estimate_peaks_from_guess(
                        bins,
                        counts,
                        guess=guesses,
                    )
                except err.DetectPeakError:
                    message = err.warn_failed_peak_detection(quad, scint)
                    logging.warning(message)
                    self._flag(quad, scint, "epeak")
                    continue

                try:
                    intervals, fit_results = self._fit_radsources_peaks(
                        bins,
                        counts,
                        limits,
                        constrains,
                    )
                except err.FailedFitError:
                    message = err.warn_failed_peak_fit(quad, scint)
                    logging.warning(message)
                    self._flag(quad, scint, "efit")
                    continue

                int_inf, int_sup = zip(*intervals)
                results.setdefault(quad, {})[scint] = np.column_stack(
                    (*fit_results, int_inf, int_sup)
                ).flatten()
        return results, radiation_sources

    @staticmethod
    def _fit_radsources_peaks(x, y, limits, constrains):
        centers, _, fwhms, _, *_ = _fit_peaks(x, y, limits)
        sigmas = fwhms / 2.35
        lower, upper = zip(*constrains)
        intervals = [*zip(centers + sigmas * lower, centers + sigmas * upper)]
        fit_results = _fit_peaks(x, y, intervals)
        return intervals, fit_results

    @as_cal_dataframe
    def _calibrate_sdds(self):
        fits = self.xfit
        energies = [s.energy for s in self._xradsources().values()]

        results = {}
        for quad in fits.keys():
            for ch in fits[quad].index:
                centers = fits[quad].loc[ch][:, "center"].values
                center_errs = fits[quad].loc[ch][:, "center_err"].values

                try:
                    cal_results = self._calibrate_chn(
                        centers,
                        energies,
                        center_errs,
                    )
                except err.FailedFitError:
                    message = err.warn_failed_linearity_fit(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "xcal")
                    continue

                results.setdefault(quad, {})[ch] = np.array(cal_results)
        return results

    @staticmethod
    def _calibrate_chn(centers, radsources: list, weights=None):
        lmod = LinearModel()
        pars = lmod.guess(centers, x=radsources)
        try:
            resultlin = lmod.fit(
                centers,
                pars,
                x=radsources,
                weights=weights,
            )
        except ValueError:
            raise err.FailedFitError("linear fitter error")

        chi2 = resultlin.redchi
        gain = resultlin.params["slope"].value
        offset = resultlin.params["intercept"].value
        gain_err = resultlin.params["slope"].stderr
        offset_err = resultlin.params["intercept"].stderr
        return gain, gain_err, offset, offset_err, chi2

    @as_slo_dataframe
    def _calibrate_scintillators(self):
        radiation_sources = self._sradsources()
        energies = [s.energy for s in radiation_sources.values()]

        results = {}
        for quad in self.efit.keys():
            df = self.efit[quad]
            for scint in df.index:
                los = df.loc[scint][:, "center"].values / energies
                lo_errs = self._scintillator_lout_error(quad, scint, energies)

                lo, lo_err = self._deal_with_multiple_gamma_decays(los, lo_errs)
                results.setdefault(quad, {})[scint] = np.array((lo, lo_err))
        return results

    def _scintillator_lout_error(self, quad, scint, energies):
        cell = scint
        cell_cal = (
            self.sdd_cal[quad]
            .loc[scint][
                [
                    "gain",
                    "gain_err",
                    "offset",
                    "offset_err",
                ]
            ]
            .to_list()
        )
        companion = self.couples[quad][cell]
        comp_cal = (
            self.sdd_cal[quad]
            .loc[companion][
                [
                    "gain",
                    "gain_err",
                    "offset",
                    "offset_err",
                ]
            ]
            .to_list()
        )

        centers_cell = self.sfit[quad].loc[cell][:, "center"].values
        centers_comp = self.sfit[quad].loc[companion][:, "center"].values
        electron_err_cell = electron_error(centers_cell, *cell_cal)
        electron_err_companion = electron_error(centers_comp, *comp_cal)
        electron_err_sum = np.sqrt(electron_err_cell**2 + electron_err_companion**2)
        fit_error = self.efit[quad].loc[cell][:, "center_err"].values
        error = (electron_err_sum + fit_error) / energies
        return error

    @as_slo_dataframe
    def _compute_effective_light_outputs(self):
        results = {}
        for quad in self.sfit.keys():
            quad_couples = self.couples[quad]
            inverted_quad_couples = {v: k for k, v in quad_couples.items()}
            for ch in self.sfit[quad].index:
                if ch in quad_couples.keys():
                    companion = quad_couples[ch]
                    scint = ch
                else:
                    companion = inverted_quad_couples[ch]
                    scint = companion

                try:
                    lo, lo_err = (
                        self.scint_cal[quad]
                        .loc[scint][
                            [
                                "light_out",
                                "light_out_err",
                            ]
                        ]
                        .to_list()
                    )

                except KeyError:
                    message = err.warn_missing_lout(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "lout")
                    continue

                else:
                    centers = self.sfit[quad].loc[ch][:, "center"].values
                    gain, offset = self.sdd_cal[quad].loc[ch][["gain", "offset"]].values
                    centers_electrons = (centers - offset) / gain / PHOTOEL_PER_KEV

                    centers_companion = (
                        self.sfit[quad].loc[companion][:, "center"].values
                    )
                    gain_comp, offset_comp = (
                        self.sdd_cal[quad].loc[companion][["gain", "offset"]].values
                    )
                    centers_electrons_comp = (
                        (centers_companion - offset_comp) / gain_comp / PHOTOEL_PER_KEV
                    )

                    effs = (
                        lo
                        * centers_electrons
                        / (centers_electrons + centers_electrons_comp)
                    )
                    eff_errs = (
                        lo_err
                        * centers_electrons
                        / (centers_electrons + centers_electrons_comp)
                    )
                    eff, eff_err = self._deal_with_multiple_gamma_decays(effs, eff_errs)
                    results.setdefault(quad, {})[ch] = np.array((eff, eff_err))
        return results

    def _effective_lout_error(self, quad, scint, ch, companion):
        # TODO: unused atm. requires further verificantions. as it is, very little
        #       differences from much simpler calculations.
        lo = self.scint_cal[quad].loc[scint]["light_out"]
        lo_err = self.scint_cal[quad].loc[scint]["light_out_err"]
        centers = self.sfit[quad].loc[ch][:, "center"].to_numpy()
        centers_err = self.sfit[quad].loc[ch][:, "center_err"].to_numpy()
        centers_companion = self.sfit[quad].loc[companion][:, "center"].to_numpy()
        centers_companion_err = (
            self.sfit[quad].loc[companion][:, "center_err"].to_numpy()
        )

        errors = np.sqrt(
            lo_err**2 * (centers / (centers + centers_companion)) ** 2
            + centers_err**2
            * (lo * centers_companion / (centers + centers_companion) ** 2) ** 2
            + centers_companion_err**2
            * (lo * centers / (centers + centers_companion) ** 2) ** 2
        )
        return errors

    @staticmethod
    def _deal_with_multiple_gamma_decays(light_outs, light_outs_errs):
        mean_lout = light_outs.mean()
        mean_lout_err = np.sqrt(np.sum(light_outs_errs**2)) / len(light_outs_errs)
        return mean_lout, mean_lout_err


def compute_histogram(value, data, bins, nthreads=1):
    def helper(quad):
        hist_quads = {}
        quad_df = data[data["QUADID"] == quad]
        for ch in range(32):
            adcs = quad_df[(quad_df["CHN"] == ch)][value]
            ys, _ = np.histogram(adcs, bins=bins)
            hist_quads[ch] = ys
        return quad, hist_quads

    results = Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in "ABCD")
    counts = {key: value for key, value in results}
    return histogram(bins, counts)


def _fit_peaks(x, y, limits):
    n_peaks = len(limits)
    centers = np.zeros(n_peaks)
    center_errs = np.zeros(n_peaks)
    fwhms = np.zeros(n_peaks)
    fwhm_errs = np.zeros(n_peaks)
    amps = np.zeros(n_peaks)
    amp_errs = np.zeros(n_peaks)

    for i in range(n_peaks):
        lowlim, hilim = limits[i]
        result, start, stop, x_fine, fitting_curve = _peak_fitter(x, y, (lowlim, hilim))
        centers[i] = result.params["center"].value
        center_errs[i] = result.params["center"].stderr
        fwhms[i] = result.params["fwhm"].value
        fwhm_errs[i] = result.params["fwhm"].stderr
        amps[i] = result.params["amplitude"].value
        amp_errs[i] = result.params["amplitude"].stderr
    return centers, center_errs, fwhms, fwhm_errs, amps, amp_errs


def _peak_fitter(x, y, limits):
    start, stop = limits
    x_start = np.where(x >= start)[0][0]
    x_stop = np.where(x < stop)[0][-1]
    x_fit = (x[x_start : x_stop + 1][1:] + x[x_start : x_stop + 1][:-1]) / 2
    y_fit = y[x_start:x_stop]
    errors = np.clip(np.sqrt(y_fit), 1, None)

    mod = GaussianModel()
    pars = mod.guess(y_fit, x=x_fit)
    try:
        result = mod.fit(y_fit, pars, x=x_fit, weights=1 / errors)
    except TypeError:
        raise err.FailedFitError("peak fitter error.")
    x_fine = np.linspace(x[0], x[-1], len(x) * 100)
    fitting_curve = mod.eval(
        x=x_fine,
        amplitude=result.best_values["amplitude"],
        center=result.best_values["center"],
        sigma=result.best_values["sigma"],
    )

    return result, start, stop, x_fine, fitting_curve


def electron_error(
    adc,
    gain,
    gain_err,
    offset,
    offset_err,
):
    error = (
        np.sqrt(
            +((offset_err / gain) ** 2) + ((adc - offset) / gain**2) * (gain_err**2)
        )
        / PHOTOEL_PER_KEV
    )
    return error
