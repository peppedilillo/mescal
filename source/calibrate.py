from bisect import bisect_right
from collections import namedtuple
import logging
from math import ceil
from math import floor

from joblib import delayed
from joblib import Parallel
from lmfit.models import GaussianModel
from lmfit.models import LinearModel
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from source.constants import PHOTOEL_PER_KEV
from source.detectors import Detector
import source.errors as err
from source.eventlist import electrons_to_energy
from source.eventlist import filter_channels
from source.eventlist import find_widows
from source.eventlist import infer_onchannels
from source.eventlist import make_electron_list
from source.eventlist import perchannel_counts
from source.io import Exporter
from source.radsources import radsources_dicts
from source.speaks import find_epeaks
from source.speaks import find_speaks
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
    "rsquared",
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


def find_adc_domain(data, maxmargin=10, roundto=500, clipquant=0.996):
    """
    find good domain limits for adc data, euristic.

    Args:
        data: array of int, adc readings
        maxmargin: int, ignores data larger than max - maxmargin
        roundto: int, round to nearest
        clipquant: float, ignore data above quantile

    Returns:

    """
    # _remove eventual zeros
    min_ = data.min()
    clipped_data = data[data > min_ + maxmargin]
    lo = clipped_data.quantile(1 - clipquant)
    lo = floor(lo / roundto) * roundto
    # _remove saturated data
    max_ = data.max()
    clipped_data = data[data < max_ - maxmargin]
    hi = clipped_data.quantile(clipquant)
    hi = ceil(hi / roundto) * roundto
    return lo, hi


def _histogram(value, data, bins, nthreads=1):
    def helper(quad):
        hist_quads = {}
        quad_df = data[data["QUADID"] == quad]
        for ch in range(32):
            adcs = quad_df[(quad_df["CHN"] == ch)][value]
            ys, _ = np.histogram(adcs, bins=bins)
            hist_quads[ch] = ys
        return quad, hist_quads

    results = Parallel(n_jobs=nthreads)(
        delayed(helper)(quad) for quad in "ABCD"
    )
    counts = {key: value for key, value in results}
    return histogram(bins, counts)


def cached_xhist():
    memory = []

    def xhistogram(data, bins, nthreads=1):
        value = "ADC"
        data = data[data["EVTYPE"] == "X"]
        histograms = _histogram(
            value,
            data,
            bins,
            nthreads=nthreads,
        )
        return histograms

    def helper(*args, **kwargs):
        if memory:
            return memory[0]
        memory.append(xhistogram(*args, **kwargs))
        return memory[0]

    return helper


xhistogram = cached_xhist()


def cached_shist():
    memory = []

    def shistogram(data, bins, nthreads=1):
        value = "ADC"
        data = data[data["EVTYPE"] == "S"]
        histograms = _histogram(
            value,
            data,
            bins,
            nthreads=nthreads,
        )
        return histograms

    def helper(*args, **kwargs):
        if memory:
            return memory[0]
        memory.append(shistogram(*args, **kwargs))
        return memory[0]

    return helper


shistogram = cached_shist()


def ehistogram(data, bins, nthreads=1):
    value = "ELECTRONS"
    data = data[data["EVTYPE"] == "S"]
    histograms = _histogram(
        value,
        data,
        bins,
        nthreads=nthreads,
    )
    return histograms


def get_ebins():
    return linrange(500, 25000, 50)


class Calibrate:
    def __init__(
        self,
        model,
        radsources,
        configuration,
        console=None,
        nthreads=1,
    ):
        self.radsources = radsources_dicts(radsources)
        self.detector = Detector(model)
        self.configuration = configuration
        self.console = console
        self.nthreads = nthreads
        self.flagged = {}
        self.eventlist = None
        self.data = None
        self._counts = {}
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
        self.xfit = None
        self.sfit = None
        self.efit = None
        self.sdd_calibration = None
        self.resolution = None
        self.scintillator_calibration = None
        self.lightoutput = None

    def __call__(self, data):
        self.data = data
        self.channels = infer_onchannels(data)
        self._bin()
        self.eventlist = self._calibrate()
        self._print_calibstatus()
        return self.eventlist

    def count(self, key="all"):
        if key in self._counts:
            return self._counts[key]
        self._counts[key] = perchannel_counts(self.data, self.channels, key=key)
        return self._counts[key]

    def get_exporter(self, filename, tabformat):
        exporter = Exporter(
            self,
            filename,
            tabformat,
            nthreads=self.nthreads,
        )
        return exporter

    def apply_nlcorrection(self, nlcorrection: pd.DataFrame):
        mask = (self.eventlist["EVTYPE"] == "S") & (
            (self.eventlist["ENERGY"] < nlcorrection["ENERGY"].iloc[0])
            | (self.eventlist["ENERGY"] >= nlcorrection["ENERGY"].iloc[-1])
        )
        self.eventlist = self.eventlist[~mask].reset_index()
        mask = self.eventlist["EVTYPE"] == "S"
        f = interp1d(nlcorrection["ENERGY"], nlcorrection["CORRFACTOR"])
        corrections = f(self.eventlist[mask]["ENERGY"])
        self.eventlist.loc[mask, "ENERGY"] /= corrections
        return

    def xradsources(self):
        xradsources, _ = self.radsources
        return xradsources

    def sradsources(self):
        _, sradsources = self.radsources
        return sradsources

    def _bin(self):
        lo, hi = find_adc_domain(self.data["ADC"])
        self.xbins = linrange(lo, hi, self.configuration["xbinning"])
        self.sbins = linrange(lo, hi, self.configuration["sbinning"])
        self.xhistograms = xhistogram(self.data, self.xbins, self.nthreads)
        self.shistograms = shistogram(self.data, self.sbins, self.nthreads)
        lost = (self.data["ADC"] >= hi) | (self.data["ADC"] < lo)
        lost_fraction = 100 * len(self.data[lost]) / len(self.data["ADC"])
        self._print(
            ":white_check_mark: Binned data. Lost {:.2f}% dataset.".format(
                lost_fraction
            )
        )

    def _calibrate_x(self):
        if len(self.xradsources()) < 2:
            raise err.FewLinesError()
        # generally self.xpeaks will not be None when
        # attempting a new sdd calibration.
        if self.xpeaks is None:
            self.xpeaks = self._detect_xpeaks()
        elif "xpeak" in self.flagged:
            for quad, ch in self.flagged["xpeak"]:
                message = err.warn_failed_peak_detection(quad, ch)
                logging.warning(message)
        self.xfit = self._fit_xradsources()
        self.sdd_calibration = self._calibrate_sdds()
        self.resolution = self._compute_energy_resolution()
        self._print(":white_check_mark: Analyzed X events.")
        return True

    def _calibrate_s(self, electron_evlist):
        if not self.sradsources():
            raise err.FewLinesError()
        if self.speaks is None:
            self.speaks = self._detect_speaks()
        elif "speak" in self.flagged:
            for quad, ch in self.flagged["speak"]:
                message = err.warn_failed_peak_detection(quad, ch)
                logging.warning(message)
        self.sfit = self._fit_sradsources()
        self.ebins = get_ebins()
        self.ehistograms = ehistogram(
            electron_evlist, self.ebins, self.nthreads
        )
        self.epeaks = self._detect_epeaks()
        self.efit = self._fit_gamma_electrons()
        self.scintillator_calibration = self._calibrate_scintillators()
        self.lightoutput = self._compute_effective_light_outputs()
        self._print(":white_check_mark: Analyzed S events.")
        return True

    def _calibrate(self):
        message = "Attempting new calibration."
        logging.info(message)

        try:
            self._calibrate_x()
        except err.FewLinesError:
            logging.warning("Too few lines to calibrate in X-mode.")
            return
        electron_evlist = make_electron_list(
            self.data,
            self.sdd_calibration,
            self.detector,
            nthreads=self.nthreads,
        )
        try:
            self._calibrate_s(electron_evlist)
        except err.FewLinesError:
            logging.warning(
                "Attempted S-calibration but got no radiation sources."
            )
            return
        try:
            eventlist = electrons_to_energy(
                electron_evlist,
                self.scintillator_calibration,
                self.detector.couples,
            )
        except err.CalibratedEventlistError as e:
            logging.warning(f"Photon events calibration error: {e}.")
            return None
        return eventlist

    def _print_calibstatus(self):
        """Prepares and exports base calibration results."""
        if not self.xradsources() and not self.sradsources():
            msg = "[bold yellow]:yellow_circle: Skipped calibration."
        elif not self.sdd_calibration and not self.lightoutput:
            msg = "[bold red]:red_circle: Calibration failed."
        elif not self.sdd_calibration or not self.lightoutput:
            msg = "[bold yellow]:yellow_circle: Calibration partially complete."
        else:
            msg = ":green_circle: Calibration complete."
        self._print(msg)

    def _print(self, message):
        if self.console:
            self.console.log(message)
        else:
            print(message)

    def _flag(self, quad, chn, flag):
        self.flagged.setdefault(flag, []).append((quad, chn))

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
                        width=ceil(5 / self.configuration["xbinning"]),
                        distance=ceil(5 / self.configuration["xbinning"]),
                        smoothing=ceil(5 / self.configuration["xbinning"]),
                        mincounts=self.configuration["xpeaks_mincounts"],
                        channel_id=(quad, ch),
                    )
                except err.DetectPeakError:
                    message = err.warn_failed_peak_detection(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "xpeak")
                    continue

                inf, sup = zip(*limits)
                results.setdefault(quad, {})[ch] = np.column_stack(
                    (inf, sup)
                ).flatten()
        return results, radiation_sources

    @as_fit_dataframe
    def _fit_xradsources(self):
        bins = self.xhistograms.bins
        radiation_sources = self.xradsources()
        constraints = [
            (s.low_lim, s.hi_lim) for s in radiation_sources.values()
        ]

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
        for quad in self.sdd_calibration.keys():
            for ch in self.sdd_calibration[quad].index:
                counts = self.shistograms.counts[quad][ch]

                try:
                    gain = (
                        self.sdd_calibration[quad].loc[ch]["gain"],
                        self.sdd_calibration[quad].loc[ch]["gain_err"],
                    )
                    offset = (
                        self.sdd_calibration[quad].loc[ch]["offset"],
                        self.sdd_calibration[quad].loc[ch]["offset_err"],
                    )
                    limits = find_speaks(
                        bins,
                        counts,
                        energies,
                        gain,
                        offset,
                        lightout_guess,
                        width=ceil(50 / self.configuration["sbinning"]),
                        distance=ceil(100 / self.configuration["sbinning"]),
                        smoothing=ceil(25 / self.configuration["sbinning"]),
                        mincounts=50,
                        channel_id=ch,
                    )

                except err.DetectPeakError:
                    companion = self.detector.companion(quad, ch)
                    if companion in self.sdd_calibration[quad].index:
                        message = err.warn_failed_peak_detection(quad, ch)
                        logging.warning(message)
                    else:
                        # companion is off and counts is empty
                        message = err.warn_widow(quad, ch, companion)
                        logging.info(message)
                    self._flag(quad, ch, "speak")
                    continue

                inf, sup = zip(*limits)
                results.setdefault(quad, {})[ch] = np.column_stack(
                    (inf, sup)
                ).flatten()
        return results, radiation_sources

    @as_fit_dataframe
    def _fit_sradsources(self):
        bins = self.shistograms.bins
        radiation_sources = self.sradsources()
        constraints = [
            (s.low_lim, s.hi_lim) for s in radiation_sources.values()
        ]

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
                    assert self.detector.scintid(
                        quad, ch
                    ) == self.detector.companion(quad, ch)
                    continue

                scint = ch
                companion = self.detector.companion(quad, ch)
                counts = self.ehistograms.counts[quad][scint]

                gain_ch = (
                    self.sdd_calibration[quad].loc[ch]["gain"],
                    self.sdd_calibration[quad].loc[ch]["gain_err"],
                )
                offset_ch = (
                    self.sdd_calibration[quad].loc[ch]["offset"],
                    self.sdd_calibration[quad].loc[ch]["offset_err"],
                )
                gain_comp = (
                    self.sdd_calibration[quad].loc[companion]["gain"],
                    self.sdd_calibration[quad].loc[companion]["gain_err"],
                )
                offset_comp = (
                    self.sdd_calibration[quad].loc[companion]["offset"],
                    self.sdd_calibration[quad].loc[companion]["offset_err"],
                )

                sfit_ch = self.sfit[quad].loc[ch][:, "center"].values
                if companion in self.sfit[quad].index:
                    sfit_comp = (
                        self.sfit[quad].loc[companion][:, "center"].values
                    )
                else:
                    # it could be that we missed a sfit.
                    # this can certainly be improve but i can't get it right now.
                    sfit_comp = sfit_ch

                try:
                    limits = find_epeaks(
                        bins,
                        counts,
                        sfit_ch,
                        sfit_comp,
                        gain_ch,
                        gain_comp,
                        offset_ch,
                        offset_comp,
                        mincounts=100,
                        width=5,
                        smoothing=10,
                        distance=20,
                        channel_id=f"{quad}{ch}",
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
        constraints = [
            (s.low_lim, s.hi_lim) for s in radiation_sources.values()
        ]

        results = {}
        for quad in self.epeaks.keys():
            for ch in self.epeaks[quad].index:
                if ch not in self.detector.couples[quad].keys():
                    assert self.detector.scintid(
                        quad, ch
                    ) == self.detector.companion(quad, ch)
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

        rss = np.sum((resultlin.data - resultlin.best_fit) ** 2)
        tss = np.sum((resultlin.data - resultlin.data.mean()) ** 2)
        try:
            rsquared = 1 - rss / tss
        except ZeroDivisionError:
            raise err.FailedFitError("cannot compute r squared")

        gain = resultlin.params["slope"].value
        offset = resultlin.params["intercept"].value
        gain_err = resultlin.params["slope"].stderr
        offset_err = resultlin.params["intercept"].stderr
        return gain, gain_err, offset, offset_err, rsquared

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

        for quad in self.sdd_calibration.keys():
            fit = self.xfit[quad]
            cal = self.sdd_calibration[quad]

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
                if (scint not in self.sfit[quad].index) or (
                    self.detector.couples[quad][scint]
                    not in self.sfit[quad].index
                ):
                    message = err.warn_failed_lightout(quad, scint)
                    logging.warning(message)
                    self._flag(quad, scint, "scal")
                    continue

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
        cell_cal = self.sdd_calibration[quad].loc[scint][params].to_list()
        companion = self.detector.couples[quad][cell]
        comp_cal = self.sdd_calibration[quad].loc[companion][params].to_list()
        centers_cell = self.sfit[quad].loc[cell][:, "center"].values
        centers_comp = self.sfit[quad].loc[companion][:, "center"].values
        electron_err_cell = self._electron_error(centers_cell, *cell_cal)
        electron_err_companion = self._electron_error(centers_comp, *comp_cal)
        electron_err_sum = np.sqrt(
            electron_err_cell**2 + electron_err_companion**2
        )
        fit_error = self.efit[quad].loc[cell][:, "center_err"].values
        error = (electron_err_sum + fit_error) / energies
        return error

    @staticmethod
    def _deal_with_multiple_gamma_decays(light_outs, light_outs_errs):
        mean_lout = light_outs.mean()
        mean_lout_err = np.sqrt(np.sum(light_outs_errs**2)) / len(
            light_outs_errs
        )
        return mean_lout, mean_lout_err

    @as_slo_dataframe
    def _compute_effective_light_outputs(self):
        results = {}
        for quad in self.sfit.keys():
            for ch in self.sfit[quad].index:
                companion = self.detector.companion(quad, ch)
                scint = self.detector.scintid(quad, ch)
                try:
                    s_ = self.scintillator_calibration[quad].loc[scint]
                    lo, lo_err = s_[["light_out", "light_out_err"]].to_list()
                except KeyError:
                    message = err.warn_missing_lout(quad, ch)
                    logging.warning(message)
                    self._flag(quad, ch, "lout")
                    continue

                else:
                    centers = self.sfit[quad].loc[ch][:, "center"].values
                    ch_ = self.sdd_calibration[quad].loc[ch]
                    gain, offset = ch_[["gain", "offset"]].values
                    centers_electrons = (
                        (centers - offset) / gain / PHOTOEL_PER_KEV
                    )
                    compfit_ = self.sfit[quad].loc[companion]
                    centers_companion = compfit_[:, "center"].values
                    comp_ = self.sdd_calibration[quad].loc[companion]
                    gain_comp, offset_comp = comp_[["gain", "offset"]].values

                    centers_electrons_comp = (
                        (centers_companion - offset_comp)
                        / gain_comp
                        / PHOTOEL_PER_KEV
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

                    eff, eff_err = self._deal_with_multiple_gamma_decays(
                        effs, eff_errs
                    )
                    results.setdefault(quad, {})[ch] = np.array((eff, eff_err))
        return results

    @staticmethod
    def _peak_fitter(x, y, limits, min_counts=100):
        start, stop = limits
        # select index of the smallest subset of x larger than [start, stop]
        x_start = max(bisect_right(x, start) - 1, 0)
        x_stop = bisect_right(x, stop)
        if x_stop - x_start < 4:
            raise err.FailedFitError("too few bins to fit.")
        x_fit = (x[x_start : x_stop + 1][1:] + x[x_start : x_stop + 1][:-1]) / 2
        y_fit = y[x_start:x_stop]
        if np.sum(y_fit) < min_counts:
            raise err.FailedFitError("too few counts to fit.")
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


@as_slo_dataframe
def _effectivelo_to_scintillatorslo(lightoutput, detector):
    results = {}
    scintillator_ids = detector.scintids()
    for quad in lightoutput.keys():
        for ch in scintillator_ids[quad]:
            if ch not in lightoutput[quad].index:
                continue
            companion = detector.companion(quad, ch)
            if companion not in lightoutput[quad].index:
                raise err.WrongTableError(
                    "Wrong light output calibration table."
                )
            ch_lo = lightoutput[quad].loc[ch].light_out
            ch_lo_err = lightoutput[quad].loc[ch].light_out_err
            companion_lo = lightoutput[quad].loc[companion].light_out
            companion_lo_err = lightoutput[quad].loc[companion].light_out_err
            lo = ch_lo + companion_lo
            lo_err = ch_lo_err + companion_lo_err
            results.setdefault(quad, {})[ch] = np.array((lo, lo_err))
    return results


class ImportedCalibration(Calibrate):
    def __init__(
        self,
        model,
        configuration,
        *ignore,
        sdd_calibration,
        lightoutput,
        **kwargs,
    ):
        if ignore:
            raise TypeError("wrong arguments. use keywords for reports.")
        super().__init__(model, [], configuration, **kwargs)
        self.sdd_calibration = sdd_calibration
        self._print(":open_book: Loaded SDD calibration.")
        self.lightoutput = lightoutput
        self._print(":open_book: Loaded scintillators calibration.")
        self.scintillator_calibration = _effectivelo_to_scintillatorslo(
            self.lightoutput, self.detector
        )

    def __call__(self, data):
        self.channels = infer_onchannels(data)
        widows = find_widows(self.channels, self.detector)
        filtered_data, waste = filter_channels(data, widows)
        self.data = filtered_data
        self._print(
            f":white_check_mark: Filtered {len(waste)} event from widow channels"
        )
        self._bin()
        self.eventlist = self._calibrate()
        return self.eventlist

    def _calibrate(self):
        electron_evlist = make_electron_list(
            self.data,
            self.sdd_calibration,
            self.detector,
            nthreads=self.nthreads,
        )
        eventlist = electrons_to_energy(
            electron_evlist,
            self.scintillator_calibration,
            self.detector.couples,
        )
        self._print("[bold green]:videocassette: Calibration complete.")
        return eventlist


class PartialCalibration(Calibrate):
    def __init__(
        self,
        model,
        configuration,
        ssources: list[str],
        sdd_calibration,
        **kwargs,
    ):
        super().__init__(model, ssources, configuration, **kwargs)
        self.sdd_calibration = sdd_calibration

    def __call__(self, data):
        self.data = data
        self.channels = infer_onchannels(data)
        self._bin()
        self.eventlist = self._calibrate()
        self._print_calibstatus()
        return self.eventlist

    def _calibrate(self):
        message = "Attempting new calibration."
        logging.info(message)

        electron_evlist = make_electron_list(
            self.data,
            self.sdd_calibration,
            self.detector,
            nthreads=self.nthreads,
        )
        try:
            self._calibrate_s(electron_evlist)
        except err.FewLinesError:
            logging.warning(
                "Attempted S-calibration but got no radiation sources."
            )
            return
        try:
            eventlist = electrons_to_energy(
                electron_evlist,
                self.scintillator_calibration,
                self.detector.couples,
            )
        except err.CalibratedEventlistError as e:
            logging.warning(f"Photon events calibration error: {e}.")
            return None
        return eventlist
