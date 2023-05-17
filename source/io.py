import logging
from math import floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits as fitsio
from joblib import Parallel, delayed

from source import paths, plot
from source.constants import PHOTOEL_PER_KEV
from source.errors import (FormatNotSupportedError, warn_nan_in_sdd_calib,
                           warn_nan_in_slo_table)


class Exporter:
    """
    A class for managing what data product can be exported.
    """

    def __init__(self, calibration, filepath, table_format, nthreads=1):
        # calibration must have been executed  already to be exported
        assert calibration.data is not None

        self.calibration = calibration
        self.writer = get_writer(table_format)
        self.filepath = filepath
        self.nthreads = nthreads

        self.can__draw_rawspectra = True

        self.can__draw_map_counts = True

        self.can__write_sdd_calibration_report = False
        if self.calibration.sdd_calibration:
            self.can__write_sdd_calibration_report = True

        self.can__write_energy_res_report = False
        if self.calibration.resolution:
            self.can__write_energy_res_report = True

        self.can__write_scintillator_calibration_report = False
        if self.calibration.scintillator_calibration:
            self.can__write_scintillator_calibration_report = True

        self.can__write_lightoutput_report = False
        if self.calibration.lightoutput:
            self.can__write_lightoutput_report = True

        self.can__write_xfit_report = False
        if self.calibration.xfit:
            self.can__write_xfit_report = True

        self.can__write_sfit_report = False
        if self.calibration.sfit:
            self.can__write_sfit_report = True

        self.can__draw_qlooks_sdd = False
        if self.calibration.sdd_calibration:
            self.can__draw_qlooks_sdd = True

        self.can__draw_qlook_scint = False
        if self.calibration.lightoutput:
            self.can__draw_qlook_scint = True

        self.can__draw_sdiagnostics = False
        if self.calibration.speaks:
            self.can__draw_sdiagnostics = True

        self.can__draw_xdiagnostic = False
        if self.calibration.xfit:
            self.can__draw_xdiagnostic = True

        self.can__draw_xspectra = False
        if self.calibration.sdd_calibration:
            self.can__draw_xspectra = True

        self.can__draw_sspectra = False
        if self.calibration.lightoutput:
            self.can__draw_sspectra = True

        self.can__draw_xspectrum = False
        if self.calibration.sdd_calibration:
            self.can__draw_xspectrum = True

        self.can__draw_sspectrum = False
        if self.calibration.lightoutput:
            self.can__draw_sspectrum = True

        self.can__draw_linearity = False
        if self.calibration.sdd_calibration:
            self.can__draw_linearity = True

        self.can__draw_map_resolution = False
        if self.calibration.resolution:
            self.can__draw_map_resolution = True

        self.can__write_eventlist = False
        if self.calibration.eventlist is not None:
            self.can__write_eventlist = True

        self.can__draw_spectrum = False
        if self.calibration.eventlist is not None:
            self.can__draw_spectrum = True

    def write_sdd_calibration_report(self):
        assert self.can__write_sdd_calibration_report
        self.writer(
            self.calibration.sdd_calibration,
            path=paths.CALREPORT(self.filepath),
        )

    def write_energy_res_report(self):
        assert self.can__write_energy_res_report
        self.writer(
            self.calibration.resolution,
            path=paths.RESREPORT(self.filepath),
        )

    def write_lightoutput_report(self):
        assert self.can__write_lightoutput_report
        self.writer(
            self.calibration.lightoutput,
            path=paths.ELOREPORT(self.filepath),
        )

    def write_scintillator_calibration_report(self):
        assert self.can__write_scintillator_calibration_report
        self.writer(
            self.calibration.scintillator_calibration,
            path=paths.SLOREPORT(self.filepath),
        )

    def write_xfit_report(self):
        assert self.can__write_xfit_report
        write_report_to_excel(
            self.calibration.xfit,
            paths.XFTREPORT(self.filepath),
        )

    def write_sfit_report(self):
        assert self.can__write_sfit_report
        write_report_to_excel(
            self.calibration.sfit,
            paths.SFTREPORT(self.filepath),
        )

    def draw_qlooks_sdd(self):
        assert self.can__draw_qlooks_sdd
        res_cal = self.calibration.sdd_calibration
        path = paths.QLKPLOT(self.filepath)

        for quad in res_cal.keys():
            quad_res_cal = res_cal[quad]
            if quad_res_cal.isnull().values.any():
                message = warn_nan_in_sdd_calib(quad)
                logging.warning(message)
                quad_res_cal = quad_res_cal.fillna(0)
            fig, axs = plot.quicklook(quad_res_cal)
            axs[0].set_title("Calibration quicklook - Quadrant {}".format(quad))
            fig.savefig(path(quad))
            plt.close(fig)

    def write_eventlist(self):
        assert self.can__write_eventlist
        write_eventlist_to_fits(
            self.calibration.eventlist,
            paths.EVLFITS(self.filepath),
        )

    def draw_qlook_scint(self):
        assert self.can__draw_qlook_scint
        res_slo = self.calibration.lightoutput
        path = paths.SLOPLOT(self.filepath)

        for quad in res_slo.keys():
            quad_res_slo = res_slo[quad]
            if quad_res_slo.isnull().values.any():
                message = warn_nan_in_slo_table(quad)
                logging.warning(message)
                quad_res_slo = quad_res_slo.fillna(0)
            fig, ax = plot.lightout(quad_res_slo)
            ax.set_title("Light output - Quadrant {}".format(quad))
            fig.savefig(path(quad))
            plt.close(fig)

    def draw_rawspectra(self):
        assert self.can__draw_rawspectra

        def helper(quad):
            for ch in range(32):
                fig, ax = plot.uncalibrated(
                    xhistograms.bins,
                    xhistograms.counts[quad][ch],
                    shistograms.bins,
                    shistograms.counts[quad][ch],
                    figsize=(8, 4.5),
                )
                ax.set_title("Uncalibrated plot - CH{:02d}Q{}".format(ch, quad))
                fig.savefig(path(quad, ch))
                plt.close(fig)

        path = paths.UNCPLOT(self.filepath)
        xhistograms = self.calibration.xhistograms
        shistograms = self.calibration.shistograms
        nthreads = self.nthreads
        return Parallel(n_jobs=nthreads)(
            delayed(helper)(quad) for quad in xhistograms.counts.keys()
        )

    def draw_sdiagnostics(self):
        assert self.can__draw_sdiagnostics

        def helper(quad):
            for ch in res_fit[quad].index:
                fig, ax = plot.diagnostics(
                    histograms.bins,
                    histograms.counts[quad][ch],
                    res_fit[quad].loc[ch].loc[:, "center"],
                    res_fit[quad].loc[ch].loc[:, "amp"],
                    res_fit[quad].loc[ch].loc[:, "fwhm"],
                    res_fit[quad]
                    .loc[ch]
                    .loc[:, ["lim_low", "lim_high"]]
                    .values.reshape(2, -1)
                    .T,
                    figsize=(8, 4.5),
                )
                ax.set_title("Diagnostic plot - CH{:02d}Q{}".format(ch, quad))
                fig.savefig(path(quad, ch))
                plt.close(fig)

        path = paths.SDNPLOT(self.filepath)
        histograms = self.calibration.shistograms
        res_fit = self.calibration.sfit
        nthreads = self.nthreads
        return Parallel(n_jobs=nthreads, max_nbytes=None)(
            delayed(helper)(quad) for quad in res_fit.keys()
        )

    def draw_xdiagnostic(self):
        assert self.can__draw_xdiagnostic

        def helper(quad):
            for ch in res_fit[quad].index:
                fig, ax = plot.diagnostics(
                    histograms.bins,
                    histograms.counts[quad][ch],
                    res_fit[quad].loc[ch].loc[:, "center"],
                    res_fit[quad].loc[ch].loc[:, "amp"],
                    res_fit[quad].loc[ch].loc[:, "fwhm"],
                    res_fit[quad]
                    .loc[ch]
                    .loc[:, ["lim_low", "lim_high"]]
                    .values.reshape(2, -1)
                    .T,
                    figsize=(8, 4.5),
                )
                ax.set_title("Diagnostic plot - CH{:02d}Q{}".format(ch, quad))
                fig.savefig(path(quad, ch))
                plt.close(fig)

        path = paths.XDNPLOT(self.filepath)
        histograms = self.calibration.xhistograms
        res_fit = self.calibration.xfit
        nthreads = self.nthreads
        return Parallel(n_jobs=nthreads, max_nbytes=None)(
            delayed(helper)(quad) for quad in res_fit.keys()
        )

    def draw_xspectra(self):
        assert self.can__draw_xspectra

        def helper(quad):
            for ch in res_cal[quad].index:
                offset = res_cal[quad].loc[ch]["offset"]
                gain = res_cal[quad].loc[ch]["gain"]
                enbins = (histograms.bins - offset) / gain
                fig, ax = plot.spectrum_x(
                    enbins,
                    histograms.counts[quad][ch],
                    radsources,
                    figsize=(8, 4.5),
                )
                ax.set_title("Spectra plot X - CH{:02d}Q{}".format(ch, quad))
                fig.savefig(path(quad, ch))
                plt.close(fig)

        path = paths.XCSPLOT(self.filepath)
        histograms = self.calibration.xhistograms
        res_cal = self.calibration.sdd_calibration
        radsources = self.calibration.xradsources()
        nthreads = self.nthreads
        return Parallel(n_jobs=nthreads)(
            delayed(helper)(quad) for quad in res_cal.keys()
        )

    def draw_sspectra(self):
        assert self.can__draw_sspectra

        def helper(quad):
            for ch in res_slo[quad].index:
                offset = res_cal[quad].loc[ch]["offset"]
                gain = res_cal[quad].loc[ch]["gain"]
                lightout = res_slo[quad]["light_out"].loc[ch]
                xenbins = (histograms.bins - offset) / gain
                enbins = xenbins / lightout / PHOTOEL_PER_KEV

                fig, ax = plot.spectrum_s(
                    enbins,
                    histograms.counts[quad][ch],
                    radsources,
                    figsize=(8, 4.5),
                )
                ax.set_title("Spectra plot S - CH{:02d}Q{}".format(ch, quad))
                fig.savefig(path(quad, ch))
                plt.close(fig)

        path = paths.SCSPLOT(self.filepath)
        histograms = self.calibration.shistograms
        res_cal = self.calibration.sdd_calibration
        res_slo = self.calibration.lightoutput
        radsources = self.calibration.sradsources()
        nthreads = self.nthreads
        return Parallel(n_jobs=nthreads)(
            delayed(helper)(quad) for quad in res_slo.keys()
        )

    def draw_spectrum(self):
        assert self.can__draw_spectrum
        self.draw_xspectrum()
        self.draw_sspectrum()
        return True

    def draw_xspectrum(self):
        assert self.can__draw_xspectrum
        path = paths.XSPPLOT(self.filepath)
        calibrated_events = self.calibration.eventlist
        radsources = self.calibration.xradsources()

        xevs = calibrated_events[calibrated_events["EVTYPE"] == "X"]
        xcounts, xbins = np.histogram(xevs["ENERGY"], bins=np.arange(2, 40, 0.05))

        fig, ax = plot.spectrum_x(
            xbins,
            xcounts,
            radsources,
            figsize=(8, 4.5),
        )
        ax.set_title("Spectrum X")
        fig.savefig(path)
        plt.close(fig)
        return True

    def draw_sspectrum(self):
        assert self.can__draw_sspectrum
        path = paths.SSPPLOT(self.filepath)
        calibrated_events = self.calibration.eventlist
        radsources = self.calibration.sradsources()

        sevs = calibrated_events[calibrated_events["EVTYPE"] == "S"]
        scounts, sbins = np.histogram(sevs["ENERGY"], bins=np.arange(30, 1000, 2))
        fig, ax = plot.spectrum_s(
            sbins,
            scounts,
            radsources,
            figsize=(8, 4.5),
        )
        ax.set_title("Spectrum S")
        fig.savefig(path)
        plt.close(fig)
        return True

    def draw_linearity(self):
        assert self.can__draw_linearity

        def helper(quad):
            for ch in res_cal[quad].index:
                fig, ax = plot.linearity(
                    *res_cal[quad].loc[ch][
                        ["gain", "gain_err", "offset", "offset_err"]
                    ],
                    res_fit[quad].loc[ch].loc[:, "center"],
                    res_fit[quad].loc[ch].loc[:, "center_err"],
                    radsources,
                    figsize=(7, 7),
                )
                ax[0].set_title("Linearity plot - CH{:02d}Q{}".format(ch, quad))
                fig.savefig(path(quad, ch))
                plt.close(fig)

        paths.LINPLOT(self.filepath)
        res_cal = self.calibration.sdd_calibration
        res_fit = self.calibration.xfit
        radsources = self.calibration.xradsources()
        nthreads = self.nthreads
        path = paths.LINPLOT(self.filepath)
        return Parallel(n_jobs=nthreads)(
            delayed(helper)(quad) for quad in res_cal.keys()
        )

    def draw_map_resolution(self):
        assert self.can__draw_map_resolution
        path = paths.RESPLOT(self.filepath)
        decays = self.calibration.xradsources()
        source = sorted(decays, key=lambda source: decays[source].energy)[0]

        fig, ax = plot.mapenres(
            source,
            self.calibration.resolution,
            self.calibration.detector.map,
        )
        fig.savefig(path)
        plt.close(fig)
        return True

    def draw_map_counts(self):
        assert self.can__draw_map_counts
        path = paths.CNTPLOT(self.filepath)
        counts = self.calibration.count()
        detmap = self.calibration.detector.map

        fig, ax = plot.mapcounts(counts, detmap)
        fig.savefig(path)
        plt.close(fig)
        return True


def get_writer(fmt):
    if fmt == "xslx":
        return write_report_to_excel
    elif fmt == "fits":
        return write_report_to_fits
    elif fmt == "csv":
        return write_report_to_csv
    else:
        raise FormatNotSupportedError("write format not supported")


def write_report_to_excel(result_df, path):
    with pd.ExcelWriter(path) as output:
        for quad in result_df.keys():
            result_df[quad].to_excel(
                output,
                sheet_name=quad,
                engine="xlsxwriter",
            )
    return True


def read_report_from_excel(from_path, kind):
    if kind == "calib":
        return pd.read_excel(from_path, index_col=0, sheet_name=None)
    elif kind == "peaks":
        return pd.read_excel(from_path, header=[0, 1], index_col=0, sheet_name=None)
    elif kind == "fits":
        return pd.read_excel(from_path, header=[0, 1], index_col=0, sheet_name=None)
    else:
        raise ValueError("kind must be either 'calib', 'peaks', or 'fits'.")


def write_report_to_fits(result_df, path):
    header = fitsio.PrimaryHDU()
    output = fitsio.HDUList([header])
    for quad in result_df.keys():
        table_quad = fitsio.BinTableHDU.from_columns(
            result_df[quad].to_records(), name="Quadrant " + quad
        )
        output.append(table_quad)
    output.writeto(path.with_suffix(".fits"), overwrite=True)
    return True


def read_report_from_fits(path):
    pass


def write_report_to_csv(result_df, path):
    for quad, df in result_df.items():
        df.to_csv(
            path.with_name(path.stem + "_quad{}".format(quad)).with_suffix(".csv")
        )
    return True


def read_report_from_csv(from_path):
    pass


def write_eventlist_to_fits(eventlist, path):
    header = fitsio.PrimaryHDU()
    output = fitsio.HDUList([header])
    table_quad = fitsio.BinTableHDU.from_columns(
        eventlist.to_records(
            index=False,
            column_dtypes={"EVTYPE": "U1", "CHN": "i8", "QUADID": "U1"},
        ),
        name="Event list",
    )
    output.append(table_quad)
    output.writeto(path, overwrite=True)
    return True


def pandas_from_LV0d5(fits: Path):
    fits_path = Path(fits)
    with fitsio.open(fits_path) as fits_file:
        fits_data = fits_file[-1].data
        df = pd.DataFrame(np.array(fits_data).byteswap().newbyteorder())
    # fixes first buffer missing ABT
    start_t = floor(df[df["TIME"] > 1].iloc[0]["TIME"]) - 1
    df.loc[df["TIME"] < 1, "TIME"] += start_t

    df = df.reset_index(level=0).rename({"index": "SID"}, axis="columns")
    columns = ["ADC", "CHN", "QUADID", "NMULT", "TIME", "SID"]
    types = ["int32", "int8", "string", "int8", "float64", "int32"]
    dtypes = {col: tp for col, tp in zip(columns, types)}

    temp = np.concatenate(
        [
            df[["ADC" + i, "CHANNEL" + i, "QUADID", "NMULT", "TIME", "SID"]]
            for i in "012345"
        ]
    )
    temp = temp[temp[:, 0] > 0]
    temp = temp[temp[:, -1].argsort()]
    df = pd.DataFrame(temp, columns=columns)
    df = df.assign(QUADID=df["QUADID"].map({0: "A", 1: "B", 2: "C", 3: "D"})).astype(
        dtypes
    )
    return df
