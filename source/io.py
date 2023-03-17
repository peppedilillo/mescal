from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits as fitsio
from joblib import Parallel, delayed
import logging

from source import paths
from source import plot
from source.errors import FormatNotSupportedError
from source.constants import PHOTOEL_PER_KEV
from source.errors import warn_nan_in_sdd_calib, warn_nan_in_slo_table


class Exporter:
    def __init__(self, calibration, filepath, table_format, nthreads=1):
        self.calibration = calibration
        self.writer = get_writer(table_format)
        self.filepath = filepath
        self.nthreads = nthreads

    def write_sdd_calibration_report(self):
        self.writer(
            self.calibration.sdd_cal,
            path=paths.CALREPORT(self.filepath),
        )

    def write_energy_res_report(self):
        self.writer(
            self.calibration.en_res,
            path=paths.RESREPORT(self.filepath),
        )

    def write_scintillator_report(self):
        self.writer(
            self.calibration.optical_coupling,
            path=paths.SLOREPORT(self.filepath),
        )

    def write_xfit_report(self):
        write_report_to_excel(
            self.calibration.xfit,
            paths.XFTREPORT(self.filepath),
        )

    def write_sfit_report(self):
        write_report_to_excel(
            self.calibration.sfit,
            paths.SFTREPORT(self.filepath),
        )

    def draw_and_save_qlooks(self):
        res_cal = self.calibration.sdd_cal
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
        return

    def draw_and_save_slo(self):
        res_slo = self.calibration.optical_coupling
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
        return

    def draw_and_save_uncalibrated(self):
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

    def draw_and_save_sdiagns(self):
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

    def draw_and_save_xdiagns(self):
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

    def draw_and_save_channels_xspectra(self):
        def helper(quad):
            for ch in res_cal[quad].index:
                enbins = (histograms.bins - res_cal[quad].loc[ch]["offset"]) / res_cal[
                    quad
                ].loc[ch]["gain"]
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
        res_cal = self.calibration.sdd_cal
        radsources = self.calibration.xradsources()
        nthreads = self.nthreads
        return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in res_cal.keys())

    def draw_and_save_channels_sspectra(self):
        def helper(quad):
            for ch in res_slo[quad].index:
                xenbins = (histograms.bins - res_cal[quad].loc[ch]["offset"]) / res_cal[
                    quad
                ].loc[ch]["gain"]
                enbins = xenbins / res_slo[quad]["light_out"].loc[ch] / PHOTOEL_PER_KEV

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
        res_cal = self.calibration.sdd_cal
        res_slo = self.calibration.optical_coupling
        radsources = self.calibration.sradsources()
        nthreads = self.nthreads
        return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in res_slo.keys())

    def draw_and_save_calibrated_spectra(self):
        self.draw_and_save_xspectrum()
        self.draw_and_save_sspectrum()
        return True

    def draw_and_save_xspectrum(self):
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

    def draw_and_save_sspectrum(self):
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

    def draw_and_save_lins(self):
        def helper(quad):
            for ch in res_cal[quad].index:
                fig, ax = plot.linearity(
                    *res_cal[quad].loc[ch][["gain", "gain_err", "offset", "offset_err"]],
                    res_fit[quad].loc[ch].loc[:, "center"],
                    res_fit[quad].loc[ch].loc[:, "center_err"],
                    radsources,
                    figsize=(7, 7),
                )
                ax[0].set_title("Linearity plot - CH{:02d}Q{}".format(ch, quad))
                fig.savefig(path(quad, ch))
                plt.close(fig)

        paths.LINPLOT(self.filepath)
        res_cal = self.calibration.sdd_cal
        res_fit = self.calibration.xfit
        radsources = self.calibration.xradsources()
        nthreads = self.nthreads
        path = paths.LINPLOT(self.filepath)
        return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in res_cal.keys())

    def draw_and_save_mapres(self):
        path = paths.RESPLOT(self.filepath)
        decays = self.calibration.xradsources()
        source = sorted(decays, key=lambda source: decays[source].energy)[0]

        fig, ax = plot.mapenres(
            source,
            self.calibration.en_res,
            self.calibration.detector.map,
        )
        fig.savefig(path)
        plt.close(fig)
        return True

    def draw_and_save_mapcounts(self):
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
                output, sheet_name=quad, engine="xlsxwriter",
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
            index=False, column_dtypes={"EVTYPE": "U1", "CHN": "i8", "QUADID": "U1"},
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

