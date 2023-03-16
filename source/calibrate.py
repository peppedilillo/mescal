import logging
from bisect import bisect_right
from collections import namedtuple
from math import ceil, floor

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lmfit.models import GaussianModel, LinearModel

import source.errors as err
from source.constants import PHOTOEL_PER_KEV
from source.eventlist import (
    add_evtype_tag,
    electrons_to_energy,
    filter_delay,
    filter_spurious,
    infer_onchannels,
    make_electron_list,
    perchannel_counts,
)
from source.speaks import find_epeaks, find_speaks
from source.xpeaks import find_xpeaks

PEAKS_PARAMS = (
    "lim_low",
    "lim_high",
)

FIT_PARAMS = (
    "center",
    "center_err",
    "fwhm",
    "fwhm_err",
    "amp",
    "amp_err",
    "lim_low",
    "lim_high",
)

CAL_PARAMS = (
    "gain",
    "gain_err",
    "offset",
    "offset_err",
    "chi2",
)

LO_PARAMS = (
    "light_out",
    "light_out_err",
)

RES_PARAMS = (
    "resolution",
    "resolution_err",
)

histogram = namedtuple("histogram", ["bins", "counts"])


def as_peaks_dataframe(f):
    def wrapper(*args):
        nested_dict, radsources = f(*args)
        quadrants = nested_dict.keys()
        index = pd.MultiIndex.from_product(
            (radsources.keys(), PEAKS_PARAMS),
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


def as_enres_dataframe(f):
    def wrapper(*args):
        nested_dict, radsources = f(*args)
        quadrants = nested_dict.keys()
        index = pd.MultiIndex.from_product(
            (radsources.keys(), RES_PARAMS),
            names=["source", "parameters"],
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


def find_adc_bins(data, binning, maxmargin=10, roundto=500, clipquant=0.996):
    """
    find good binning for adc data, euristic.
    not intended to use for binning scintillator electrons data.

    Args:
        data: array of int, adc readings
        binning: int, binning step
        maxmargin: int, ignores data larger than max - maxmargin
        roundto: int, round to nearest
        clipquant: float, ignore data above quantile

    Returns:

    """
    # _remove eventual zeros
    min = data.min()
    clipped_data = data[data > min + maxmargin]
    lo = clipped_data.quantile(1 - clipquant)
    lo = floor(lo / roundto) * roundto
    # _remove saturated data
    max = data.max()
    clipped_data = data[data < max - maxmargin]
    hi = clipped_data.quantile(clipquant)
    hi = ceil(hi / roundto) * roundto

    bins = linrange(lo, hi, binning)
    return bins


class Calibrate:
    def __init__(
        self,
        detector,
        radsources,
        configuration,
        console=None,
        nthreads=1,
    ):
        self.radsources = radsources
        self.detector = detector
        self.configuration = configuration
        self._counts = None
        self.data = None
        self.channels = None
        self.xbins = None
        self.sbins = None
        self.ebins = None
        self.xhistograms = None
        self.shistograms = None
        self.ehistograms = None
        self.xpeaks = None
        self.speaks = None
        self.epeaks = None
        self.xfit = {}
        self.sfit = {}
        self.efit = {}
        self.sdd_cal = {}
        self.en_res = {}
        self.scint_cal = {}
        self.optical_coupling = {}
        self.flagged = {}
        self.console = console
        self.nthreads = nthreads

    def __call__(self, data):
        self.channels = infer_onchannels(data)
        self.data = self._preprocess(data)
        self._bin()
        eventlist = self._calibrate()
        return eventlist

    def _bin(self):
        binning = self.configuration["binning"]

        bins = find_adc_bins(self.data["ADC"], binning)
        self.xbins = bins
        self.sbins = bins
        self.xhistograms = self._make_xhistograms(self.data)
        self.shistograms = self._make_shistograms(self.data)
        lost = (self.data["ADC"] >= bins[-1]) | (self.data["ADC"] < bins[0])
        lost_fraction = 100 * len(self.data[lost]) / len(self.data["ADC"])
        self._print(
            ":white_check_mark: Binned data. Lost {:.2f}% dataset.".format(
                lost_fraction
            )
        )

    def _preprocess(
        self,
        data,
    ):
        spurious_bool = self.configuration["filter_spurious"]
        retrigger_delay = self.configuration["filter_retrigger"]

        data = add_evtype_tag(data, self.detector.couples)
        events_pre_filter = len(data)
        self.console.log(":white_check_mark: Tagged X and S events.")

        if retrigger_delay:
            data = filter_delay(data, retrigger_delay)
        else:
            self.console.log(":exclamation_mark: Retrigger filter is off.")
        if spurious_bool:
            data = filter_spurious(data)
        else:
            self.console.log(":exclamation_mark: Spurious filter is off.")

        filtered = 100 * (events_pre_filter - len(data)) / events_pre_filter
        if filtered:
            self.console.log(
                ":white_check_mark: Filtered {:.1f}% of the events.".format(filtered)
            )
        return data

    def _calibrate(self):
        message = "attempting new calibration."
        logging.info(message)

        # X calibration
        if len(self.xradsources()) < 2:
            return None
        # generally self.xpeaks will not be None when
        # attempting a new sdd calibration.
        if self.xpeaks is None:
            self.xpeaks = self._detect_xpeaks()
        elif "xpeak" in self.flagged:
            for quad, ch in self.flagged["xpeak"]:
                message = err.warn_failed_peak_detection(quad, ch)
                logging.warning(message)
        self.xfit = self._fit_xradsources()
        self.sdd_cal = self._calibrate_sdds()
        self.en_res = self._compute_energy_resolution()
        self._print(":white_check_mark: Analyzed X events.")

        # S calibration
        if not self.sradsources():
            return None
        self.speaks = self._detect_speaks()
        self.sfit = self._fit_sradsources()
        electron_evlist = make_electron_list(
            self.data,
            self.sdd_cal,
            self.sfit,
            self.detector.couples,
            self.nthreads,
        )
        self.ebins = linrange(1000, 25000, 50)
        self.ehistograms = self._make_ehistograms(electron_evlist)
        self.epeaks = self._detect_epeaks()
        self.efit = self._fit_gamma_electrons()
        self.scint_cal = self._calibrate_scintillators()
        self.optical_coupling = self._compute_effective_light_outputs()
        self._print(":white_check_mark: Analyzed S events.")

        if not self.scint_cal:
            return None
        try:
            eventlist = electrons_to_energy(
                electron_evlist, self.scint_cal, self.detector.couples
            )
        except err.CalibratedEventlistError:
            logging.warning("Event list creation failed.")
            return None
        return eventlist

    def _print(self, message):
        if self.console:
            self.console.log(message)
        else:
            print(message)

    def xradsources(self):
        xradsources, _ = self.radsources
        return xradsources

    def sradsources(self):
        _, sradsources = self.radsources
        return sradsources

    # TODO: to be later moved to a dedicated assembly model object
    def _companion(self, quad, ch):
        """
        given a channel's quadrant and index,
        returns the id of the companion cell

        Args:
            quad: string, quadrant id
            ch: int, channel id

        Returns: int, companion channel id

        """
        if ch in self.detector.couples[quad].keys():
            return self.detector.couples[quad][ch]
        else:
            companions = {k: v for v, k in self.detector.couples[quad].items()}
            return companions[ch]

    # TODO: to be later moved to a dedicated assembly model object
    def _scintid(self, quad, ch):
        """
        given a channel's quadrant and index,
        returns the scintillator id.

        Args:
            quad: string, quadrant id
            ch: int, channel id

        Returns: int, scintillator id

        """
        if ch in self.detector.couples[quad].keys():
            return ch
        else:
            return self._companion(quad, ch)

    def _flag(self, quad, chn, flag):
        self.flagged.setdefault(flag, []).append((quad, chn))

    @staticmethod
    def _histogram(value, data, bins, nthreads=1):
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

    def _make_xhistograms(self, data):
        value = "ADC"
        bins = self.xbins
        data = data[data["EVTYPE"] == "X"]
        histograms = self._histogram(
            value,
            data,
            bins,
            nthreads=self.nthreads,
        )
        return histograms

    def _make_shistograms(self, data):
        value = "ADC"
        bins = self.sbins
        data = data[data["EVTYPE"] == "S"]
        histograms = self._histogram(
            value,
            data,
            bins,
            nthreads=self.nthreads,
        )
        return histograms

    def _make_ehistograms(self, electron_evlist):
        value = "ELECTRONS"
        data = electron_evlist[electron_evlist["EVTYPE"] == "S"]
        bins = self.ebins
        histograms = self._histogram(
            value,
            data,
            bins,
            nthreads=self.nthreads,
        )
        return histograms

    @as_peaks_dataframe
    def _detect_xpeaks(self):
        gain_center = self.configuration["gain_center"]
        gain_sigma = self.configuration["gain_sigma"]
        offset_center = self.configuration["offset_center"]
        offset_sigma = self.configuration["offset_sigma"]
        gain_guess = (gain_center, gain_sigma)  # peak detection prior estimate
        offset_guess = (
            offset_center,
            offset_sigma,
        )  # peak detection prior estimate

        bins = self.xhistograms.bins
        radiation_sources = self.xradsources()
        energies = [s.energy for s in radiation_sources.values()]

        results = {}
        for quad in self.channels.keys():
            for ch in self.channels[quad]:
                counts = self.xhistograms.counts[quad][ch]

                try:
                    limits = find_xpeaks(
                        bins,
                        counts,
                        energies,
                        gain_guess,
                        offset_guess,
                        mincounts=self.configuration["xpeaks_mincounts"],
                        channel_id=(quad, ch),
                    )
                except err.DetectPeakError:
                    message = err.warn_failed_peak_detection(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "xpeak")
                    continue

                inf, sup = zip(*limits)
                results.setdefault(quad, {})[ch] = np.column_stack((inf, sup)).flatten()
        return results, radiation_sources

    @as_fit_dataframe
    def _fit_xradsources(self):
        bins = self.xhistograms.bins
        radiation_sources = self.xradsources()
        constraints = [(s.low_lim, s.hi_lim) for s in radiation_sources.values()]

        results = {}
        for quad in self.xpeaks.keys():
            for ch in self.xpeaks[quad].index:
                row = self.xpeaks[quad].loc[ch]
                counts = self.xhistograms.counts[quad][ch]
                limits = [
                    (row[source]["lim_low"], row[source]["lim_high"])
                    for source in radiation_sources.keys()
                ]

                try:
                    intervals, fit_results = self._fit_gaussians_to_peaks(
                        bins,
                        counts,
                        limits,
                        constraints,
                    )
                except err.FailedFitError:
                    message = err.warn_failed_peak_fit(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "xfit")
                    continue

                int_inf, int_sup = zip(*intervals)
                results.setdefault(quad, {})[ch] = np.column_stack(
                    (*fit_results, int_inf, int_sup)
                ).flatten()
        return results, radiation_sources

    @as_peaks_dataframe
    def _detect_speaks(self):
        lightout_center = self.configuration["lightout_center"]
        lightout_sigma = self.configuration["lightout_sigma"]
        lightout_guess = (lightout_center, lightout_sigma)

        bins = self.shistograms.bins
        radiation_sources = self.sradsources()
        energies = [s.energy for s in radiation_sources.values()]

        results = {}
        for quad in self.sdd_cal.keys():
            for ch in self.sdd_cal[quad].index:
                counts = self.shistograms.counts[quad][ch]
                gain = self.sdd_cal[quad].loc[ch]["gain"]
                offset = self.sdd_cal[quad].loc[ch]["offset"]

                try:
                    limits = find_speaks(
                        bins,
                        counts,
                        energies,
                        gain,
                        offset,
                        lightout_guess,
                    )
                except err.DetectPeakError:
                    companion = self._companion(quad, ch)
                    if companion in self.sdd_cal[quad].index:
                        message = err.warn_failed_peak_detection(quad, ch)
                        logging.warning(message)
                    else:
                        # companion is off and counts is empty
                        message = err.warn_widow(quad, ch, companion)
                        logging.info(message)
                    self._flag(quad, ch, "speak")
                    continue

                inf, sup = zip(*limits)
                results.setdefault(quad, {})[ch] = np.column_stack((inf, sup)).flatten()
        return results, radiation_sources

    @as_fit_dataframe
    def _fit_sradsources(self):
        bins = self.shistograms.bins
        radiation_sources = self.sradsources()
        constraints = [(s.low_lim, s.hi_lim) for s in radiation_sources.values()]

        results = {}
        for quad in self.speaks.keys():
            for ch in self.speaks[quad].index:
                counts = self.shistograms.counts[quad][ch]
                row = self.speaks[quad].loc[ch]
                limits = [
                    (row[source]["lim_low"], row[source]["lim_high"])
                    for source in radiation_sources.keys()
                ]

                try:
                    intervals, fit_results = self._fit_gaussians_to_peaks(
                        bins,
                        counts,
                        limits,
                        constraints,
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

    @as_peaks_dataframe
    def _detect_epeaks(self):
        lightout_center = self.configuration["lightout_center"]
        lightout_sigma = self.configuration["lightout_sigma"]
        lightout_guess = (lightout_center, lightout_sigma)

        bins = self.ehistograms.bins
        radiation_sources = self.sradsources()
        energies = [s.energy for s in radiation_sources.values()]

        results = {}
        for quad in self.sfit.keys():
            for ch in self.sfit[quad].index:
                if ch not in self.detector.couples[quad].keys():
                    assert self._scintid(quad, ch) == self._companion(quad, ch)
                    continue
                scint = ch
                counts = self.ehistograms.counts[quad][scint]

                try:
                    limits = find_epeaks(
                        bins,
                        counts,
                        energies,
                        lightout_guess,
                    )
                except err.DetectPeakError:
                    message = err.warn_failed_peak_detection(quad, scint)
                    logging.warning(message)
                    self._flag(quad, scint, "epeaks")
                    continue

                inf, sup = zip(*limits)
                results.setdefault(quad, {})[scint] = np.column_stack(
                    (inf, sup)
                ).flatten()
        return results, radiation_sources

    @as_fit_dataframe
    def _fit_gamma_electrons(self):
        bins = self.ehistograms.bins
        radiation_sources = self.sradsources()
        constraints = [(s.low_lim, s.hi_lim) for s in radiation_sources.values()]

        results = {}
        for quad in self.epeaks.keys():
            for ch in self.epeaks[quad].index:
                if ch not in self.detector.couples[quad].keys():
                    assert self._scintid(quad, ch) == self._companion(quad, ch)
                    continue
                scint = ch
                counts = self.ehistograms.counts[quad][scint]
                row = self.epeaks[quad].loc[ch]
                limits = [
                    (row[source]["lim_low"], row[source]["lim_high"])
                    for source in radiation_sources.keys()
                ]

                try:
                    intervals, fit_results = self._fit_gaussians_to_peaks(
                        bins,
                        counts,
                        limits,
                        constraints,
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
    def _calibrate_chn(centers, energies: list, weights=None):
        lmod = LinearModel()
        pars = lmod.guess(centers, x=energies)
        try:
            resultlin = lmod.fit(
                centers,
                pars,
                x=energies,
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

    @as_cal_dataframe
    def _calibrate_sdds(self):
        fits = self.xfit
        energies = [s.energy for s in self.xradsources().values()]

        results = {}
        for quad in fits.keys():
            for ch in fits[quad].index:
                centers = fits[quad].loc[ch][:, "center"].values
                center_errs = fits[quad].loc[ch][:, "center_err"].values

                try:
                    cal_results = self._calibrate_chn(
                        centers,
                        energies,
                        weights=1 / center_errs**2,
                    )
                except err.FailedFitError:
                    message = err.warn_failed_linearity_fit(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "xcal")
                    continue

                results.setdefault(quad, {})[ch] = np.array(cal_results)
        return results

    @as_enres_dataframe
    def _compute_energy_resolution(self):
        results = {}
        radiation_sources = self.xradsources()

        for quad in self.sdd_cal.keys():
            fit = self.xfit[quad]
            cal = self.sdd_cal[quad]

            assert np.isin(cal.index, fit.index).all()
            for ch in cal.index:
                helper = []
                for source, decay in radiation_sources.items():
                    fwhms = fit[source]["fwhm"].loc[ch]
                    fwhms_err = fit[source]["fwhm_err"].loc[ch]
                    gains = cal["gain"].loc[ch]
                    gains_err = cal["gain_err"].loc[ch]

                    energyres = fwhms / gains
                    energyres_err = energyres * np.sqrt(
                        (fwhms_err / fwhms) ** 2 + (gains_err / gains) ** 2
                    )
                    helper.append((energyres, energyres_err))

                results.setdefault(quad, {})[ch] = np.hstack(helper)
        return results, radiation_sources

    @as_slo_dataframe
    def _calibrate_scintillators(self):
        radiation_sources = self.sradsources()
        energies = [s.energy for s in radiation_sources.values()]

        results = {}
        for quad in self.efit.keys():
            for scint in self.efit[quad].index:
                los = self.efit[quad].loc[scint][:, "center"].values / energies
                lo_errs = self._scintillator_lout_error(quad, scint, energies)

                lo, lo_err = self._deal_with_multiple_gamma_decays(los, lo_errs)
                results.setdefault(quad, {})[scint] = np.array((lo, lo_err))
        return results

    @staticmethod
    def _electron_error(
        adc,
        gain,
        gain_err,
        offset,
        offset_err,
    ):
        error = (
            np.sqrt(
                +((offset_err / gain) ** 2)
                + ((adc - offset) / gain**2) * (gain_err**2)
            )
            / PHOTOEL_PER_KEV
        )
        return error

    def _scintillator_lout_error(self, quad, scint, energies):
        cell = scint
        params = [
            "gain",
            "gain_err",
            "offset",
            "offset_err",
        ]
        cell_cal = self.sdd_cal[quad].loc[scint][params].to_list()
        companion = self.detector.couples[quad][cell]
        comp_cal = self.sdd_cal[quad].loc[companion][params].to_list()

        centers_cell = self.sfit[quad].loc[cell][:, "center"].values
        centers_comp = self.sfit[quad].loc[companion][:, "center"].values
        electron_err_cell = self._electron_error(centers_cell, *cell_cal)
        electron_err_companion = self._electron_error(centers_comp, *comp_cal)
        electron_err_sum = np.sqrt(electron_err_cell**2 + electron_err_companion**2)
        fit_error = self.efit[quad].loc[cell][:, "center_err"].values
        error = (electron_err_sum + fit_error) / energies
        return error

    @staticmethod
    def _deal_with_multiple_gamma_decays(light_outs, light_outs_errs):
        mean_lout = light_outs.mean()
        mean_lout_err = np.sqrt(np.sum(light_outs_errs**2)) / len(light_outs_errs)
        return mean_lout, mean_lout_err

    @as_slo_dataframe
    def _compute_effective_light_outputs(self):
        results = {}
        for quad in self.sfit.keys():
            quad_couples = self.detector.couples[quad]
            for ch in self.sfit[quad].index:
                companion = self._companion(quad, ch)
                scint = self._scintid(quad, ch)

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

    @staticmethod
    def _peak_fitter(x, y, limits):
        start, stop = limits
        # select index of the smallest subset of x larger than [start, stop]
        x_start = max(bisect_right(x, start) - 1, 0)
        x_stop = bisect_right(x, stop)
        if x_stop - x_start < 5:
            raise err.FailedFitError("too few bins to fit.")
        x_fit = (x[x_start : x_stop + 1][1:] + x[x_start : x_stop + 1][:-1]) / 2
        y_fit = y[x_start:x_stop]
        errors = np.clip(np.sqrt(y_fit), 1, None)

        mod = GaussianModel()
        mod.set_param_hint("center", min=start, max=stop)
        mod.set_param_hint("height", value=max(y_fit))
        mod.set_param_hint("sigma", max=stop - start)
        pars = mod.guess(y_fit, x=x_fit)
        result = mod.fit(
            y_fit,
            pars,
            x=x_fit,
            weights=1 / errors**2,
        )

        x_fine = np.linspace(x[0], x[-1], len(x) * 100)
        fitting_curve = mod.eval(
            x=x_fine,
            amplitude=result.best_values["amplitude"],
            center=result.best_values["center"],
            sigma=result.best_values["sigma"],
        )

        return result, start, stop, x_fine, fitting_curve

    def _fit_peaks(self, x, y, limits):
        n_peaks = len(limits)
        centers = np.zeros(n_peaks)
        center_errs = np.zeros(n_peaks)
        fwhms = np.zeros(n_peaks)
        fwhm_errs = np.zeros(n_peaks)
        amps = np.zeros(n_peaks)
        amp_errs = np.zeros(n_peaks)

        for i in range(n_peaks):
            lowlim, hilim = limits[i]
            result, start, stop, x_fine, fitting_curve = self._peak_fitter(
                x, y, (lowlim, hilim)
            )
            centers[i] = result.params["center"].value
            center_errs[i] = result.params["center"].stderr
            fwhms[i] = result.params["fwhm"].value
            fwhm_errs[i] = result.params["fwhm"].stderr
            amps[i] = result.params["amplitude"].value
            amp_errs[i] = result.params["amplitude"].stderr
        return centers, center_errs, fwhms, fwhm_errs, amps, amp_errs

    def _fit_gaussians_to_peaks(self, x, y, limits, constraints):
        centers, _, fwhms, _, *_ = self._fit_peaks(x, y, limits)
        sigmas = fwhms / 2.35
        lower, upper = zip(*constraints)
        intervals = [*zip(centers + sigmas * lower, centers + sigmas * upper)]
        fit_results = self._fit_peaks(x, y, intervals)
        return intervals, fit_results

    def counts(self):
        if self._counts is None:
            self._counts = perchannel_counts(self.data, self.channels)
        return self._counts
