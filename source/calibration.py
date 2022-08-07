import logging

import numpy as np
import pandas as pd
from lmfit.models import LinearModel

import source.errors as err
from source.constants import PHOTOEL_PER_KEV
from source.eventlist import electrons_to_energy, make_electron_list
from source.inventory import fetch_default_sdd_calibration
from source.speaks import _estimate_peaks_from_guess
from source.specutils import _fit_peaks, compute_histogram
from source.xpeaks import _find_peaks_limits

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
        self.xhistograms = self.make_xistograms(data)
        self.shistograms = self.make_shistograms(data)
        self.print(":white_check_mark: Binned data.")

        if self.get_x_radsources():
            self.xfit = self.fit_xradsources()
            self.sdd_cal = self.calibrate_sdds()
            self.print(":white_check_mark: Analyzed X events.")
            if self.get_gamma_radsources():
                self.sfit = self.fit_sradsources()
                electron_evlist = make_electron_list(
                    data,
                    self.sdd_cal,
                    self.sfit,
                    self.couples,
                )
                self.print(":white_check_mark: Made electron list")
                self.ehistograms = self.make_ehistograms(electron_evlist)
                self.efit = self.fit_gamma_electrons()
                self.scint_cal = self.calibrate_scintillators()
                self.optical_coupling = self.compute_effective_light_outputs()
                self.print(":white_check_mark: Analyzed gamma events.")
                eventlist = electrons_to_energy(
                    electron_evlist, self.scint_cal, self.couples
                )

                return eventlist

    def print(self, message):
        if self.console:
            self.console.log(message)
        else:
            print(message)

    def get_x_radsources(self):
        xradsources, _ = self.radsources
        return xradsources

    def get_gamma_radsources(self):
        _, sradsources = self.radsources
        return sradsources

    def flag_channel(self, quad, chn, flag):
        self.flagged.setdefault(flag, []).append((quad, chn))

    def fetch_hint(self):
        try:
            hint, key = fetch_default_sdd_calibration(
                self.detector_model,
                self.temperature,
            )
        except err.DetectorModelNotFound:
            hint = None
        else:
            self.print(":open_book: Loaded detection hints for {}@{}°C.".format(*key))
        return hint

    def make_xistograms(self, data):
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

    def make_shistograms(self, data):
        value = "ADC"
        bins = self.adc_bins
        data = data[data["EVTYPE"] == "S"]
        histograms = compute_histogram(
            value,
            data,
            bins,
        )
        return histograms

    def make_ehistograms(self, electron_evlist):
        value = "ELECTRONS"
        data = electron_evlist[electron_evlist["EVTYPE"] == "S"]
        bins = self.electron_bins
        histograms = compute_histogram(
            value,
            data,
            bins,
        )
        return histograms

    @as_fit_dataframe
    def fit_xradsources(self):
        bins = self.xhistograms.bins
        hint = self.fetch_hint()
        radiation_sources = self.get_x_radsources()
        energies = [s.energy for s in radiation_sources.values()]
        constrains = [(s.low_lim, s.hi_lim) for s in radiation_sources.values()]

        results = {}
        for quad in self.channels.keys():
            for ch in self.channels[quad]:
                counts = self.xhistograms.counts[quad][ch]

                try:

                    def packaged_hint():
                        return hint[quad].loc[ch] if hint else False

                    limits = _find_peaks_limits(
                        bins,
                        counts,
                        energies,
                        packaged_hint,
                    )

                except err.DetectPeakError:
                    message = err.warn_failed_peak_detection(quad, ch)
                    logging.warning(message)
                    self.flag_channel(quad, ch, "xpeak")
                    continue

                try:
                    intervals, fit_results = self.fit_radsources_peaks(
                        bins,
                        counts,
                        limits,
                        constrains,
                    )
                except err.FailedFitError:
                    meassage = err.warn_failed_peak_fit(quad, ch)
                    logging.warning(meassage)
                    self.flag_channel(quad, ch, "xfit")
                    continue

                int_inf, int_sup = zip(*intervals)
                results.setdefault(quad, {})[ch] = np.column_stack(
                    (*fit_results, int_inf, int_sup)
                ).flatten()
        return results, radiation_sources

    @as_fit_dataframe
    def fit_sradsources(self):
        bins = self.shistograms.bins
        cal_df = self.sdd_cal
        radiation_sources = self.get_gamma_radsources()
        energies = [s.energy for s in radiation_sources.values()]
        constrains = [(s.low_lim, s.hi_lim) for s in radiation_sources.values()]

        results = {}
        for quad in cal_df.keys():
            for ch in cal_df[quad].index:
                counts = self.shistograms.counts[quad][ch]
                gain = cal_df[quad].loc[ch]["gain"]
                offset = cal_df[quad].loc[ch]["offset"]

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
                    self.flag_channel(quad, ch, "speak")
                    continue

                try:
                    intervals, fit_results = self.fit_radsources_peaks(
                        bins,
                        counts,
                        limits,
                        constrains,
                    )
                except err.FailedFitError:
                    message = err.warn_failed_peak_fit(quad, ch)
                    logging.warning(message)
                    self.flag_channel(quad, ch, "sfit")
                    continue

                int_inf, int_sup = zip(*intervals)
                results.setdefault(quad, {})[ch] = np.column_stack(
                    (*fit_results, int_inf, int_sup)
                ).flatten()
        return results, radiation_sources

    @as_fit_dataframe
    def fit_gamma_electrons(self):
        bins = self.ehistograms.bins
        radiation_sources = self.get_gamma_radsources()
        energies = [s.energy for s in radiation_sources.values()]
        constrains = [(s.low_lim, s.hi_lim) for s in radiation_sources.values()]
        guesses = [
            [lout_lim * lv for lout_lim in self.light_out_guess] for lv in energies
        ]

        results = {}
        for quad in self.couples.keys():
            for ch in self.couples[quad].values():
                counts = self.ehistograms.counts[quad][ch]
                try:
                    limits = _estimate_peaks_from_guess(
                        bins,
                        counts,
                        guess=guesses,
                    )
                except err.DetectPeakError:
                    message = err.warn_failed_peak_detection(quad, ch)
                    logging.warning(message)
                    self.flag_channel(quad, ch, "epeak")
                    continue

                try:
                    intervals, fit_results = self.fit_radsources_peaks(
                        bins,
                        counts,
                        limits,
                        constrains,
                    )
                except err.FailedFitError:
                    message = err.warn_failed_peak_fit(quad, ch)
                    logging.warning(message)
                    self.flag_channel(quad, ch, "efit")
                    continue

                int_inf, int_sup = zip(*intervals)
                results.setdefault(quad, {})[ch] = np.column_stack(
                    (*fit_results, int_inf, int_sup)
                ).flatten()
        return results, radiation_sources

    @staticmethod
    def fit_radsources_peaks(x, y, limits, constrains):
        centers, _, fwhms, _, *_ = _fit_peaks(x, y, limits)
        sigmas = fwhms / 2.35
        lower, upper = zip(*constrains)
        intervals = [*zip(centers + sigmas * lower, centers + sigmas * upper)]
        fit_results = _fit_peaks(x, y, intervals)
        return intervals, fit_results

    @as_cal_dataframe
    def calibrate_sdds(self):
        fits = self.xfit
        energies = [s.energy for s in self.get_x_radsources().values()]

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
                    self.flag_channel(quad, ch, "xcal")
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
    def calibrate_scintillators(self):
        radiation_sources = self.get_gamma_radsources()
        energies = [s.energy for s in radiation_sources.values()]

        results = {}
        for quad in self.efit.keys():
            df = self.efit[quad]
            for ch in df.index:
                los = df.loc[ch][:, "center"].values / energies
                lo_errs = df.loc[ch][:, "center_err"].values / energies

                lo, lo_err = self.deal_with_multiple_gamma_decays(los, lo_errs)
                results.setdefault(quad, {})[ch] = np.array((lo, lo_err))
        return results

    @as_slo_dataframe
    def compute_effective_light_outputs(self):
        sfit_results = self.sfit
        scint_calibs = self.scint_cal
        couples = self.couples

        results = {}
        for quad in sfit_results.keys():
            sfit = sfit_results[quad]
            scint = scint_calibs[quad]
            quad_couples = couples[quad]
            inverted_quad_couples = {v: k for k, v in quad_couples.items()}
            for ch in sfit.index:
                try:
                    companion = quad_couples[ch]
                except KeyError:
                    companion = inverted_quad_couples[ch]

                if ch in scint.index:
                    lo = scint.loc[ch]["light_out"]
                    lo_err = scint.loc[ch]["light_out_err"]
                else:
                    lo = scint.loc[companion]["light_out"]
                    lo_err = scint.loc[companion]["light_out_err"]

                centers = sfit.loc[ch][:, "center"].values
                centers_companion = sfit.loc[companion][:, "center"].values
                effs = lo * centers / (centers + centers_companion)
                eff_errs = lo_err * centers / (centers + centers_companion)
                eff, eff_err = self.deal_with_multiple_gamma_decays(effs, eff_errs)
                results.setdefault(quad, {})[ch] = np.array((eff, eff_err))
        return results

    @staticmethod
    def deal_with_multiple_gamma_decays(light_outs, light_outs_errs):
        return light_outs.mean(), np.sqrt(np.sum(light_outs_errs**2))
