from collections import namedtuple
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import source.errors as err
from source.inventory import fetch_default_sdd_calibration
from source.xpeaks import _find_peaks_limits

histogram = namedtuple('histogram', ['bins', 'counts'])


def as_dict_of_dataframes(f):
    def wrapper(*args):
        nested_dict, radsources, flagged = f(*args)
        quadrants = nested_dict.keys()
        index = pd.MultiIndex.from_product(
            (radsources.keys(), FIT_PARAMS),
            names=['source', 'parameter']
        )

        dict_of_dfs = {
            q: pd.DataFrame(
                nested_dict[q],
                index=index
            ).T.rename_axis("channel")
            for q in quadrants
        }
        return dict_of_dfs, flagged
    return wrapper


def adc_bins():
    start, stop, step = 15000, 28000, 10
    nbins = int((stop - start) / step)
    end = start + nbins * step
    bins = np.linspace(start, end, nbins + 1)
    return bins


def compute_histogram(value, data, bins, nthreads=1):
    def helper(quad):
        hist_quads = {}
        quad_df = data[data['QUADID'] == quad]
        for ch in range(32):
            adcs = quad_df[(quad_df['CHN'] == ch)][value]
            ys, _ = np.histogram(adcs, range=(start, end), bins=nbins)
            hist_quads[ch] = ys
        return quad, hist_quads

    results = Parallel(n_jobs=nthreads)(delayed(helper)(quad)
                                        for quad in 'ABCD')
    counts = {key: value for key, value in results}
    return histogram(bins, counts)


def move_mean(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()


class Calibration:
    def __init__(
            self,
            channels,
            couples,
            radsources,
            detector_model=None,
            temperature=None,
            console=None,
    ):
        self.console = console
        self.radsources = radsources
        self.channels = channels
        self.couples = couples
        self.detector_model = detector_model
        self.temperature = temperature

    def __call__(self, data):
        adc_bins = adc_bins()
        x_data = data[data["EVTYPE"] == "X"]
        s_data = data[data["EVTYPE"] == "S"]

        self.xhistograms = compute_histogram(
            'ADC',
            x_data,
            adc_bins
        )
        self.shistograms = compute_histogram(
            'ADC',
            s_data,
            adc_bins
        )
        self.print(
            ":white_check_mark: Binned data."
        )

        xradsources = self.get_x_radsources()
        if xradsources:
            xfit_results, _ = self.fit_xradsources()

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

    def fetch_hint(self):
        try:
            hint, key = fetch_default_sdd_calibration(
                self.model,
                self.temperature
            )
        except err.DetectorModelNotFound:
            hint = None
        else:
            self.print(
                ":open_book: Loaded detection hints for {}@{}°C."
                .format(*key)
            )
        return hint

    @as_dict_of_dataframes
    def fit_xradsources(self):
        results, flagged = {}, {}
        energies = [s.energy for s in self.get_x_radsources().values()]
        hint = self.fetch_hint()

        for quad in self.channels.keys():
            for ch in self.channels[quad]:
                bins = self.xhistograms.bins
                counts = self.xhistograms.counts[quad][ch]

                try:
                    def packaged_hint():
                        return hint[quad].loc[ch] if hint else None
                    limits = _find_peaks_limits(
                        bins,
                        counts,
                        energies,
                        packaged_hint,
                    )

                except err.DetectPeakError:
                    message = err.warn_failed_peak_detection(quad, ch)
                    logging.warning(message)
                    flagged.setdefault(quad, []).append(ch)
                    continue

                try:
                    intervals, fit_results = self.fit_radsources_peaks(
                        bins,
                        counts,
                        limits,
                    )
                except err.FailedFitError:
                    meassage = warn_failed_peak_fit(quad, ch)
                    logging.warning(meassage)
                    flagged.setdefault(quad, []).append(ch)
                    continue

                int_inf, int_sup = zip(*intervals)
                results.setdefault(quad, {})[ch] = np.column_stack(
                    (*fit_results, int_inf, int_sup)).flatten()
        return results, flagged


