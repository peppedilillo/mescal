import logging
from math import pi, sqrt

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import source.fcparams as fcm
from source.constants import PHOTOEL_PER_KEV
from source.errors import warn_nan_in_sdd_calib, warn_nan_in_slo_table

matplotlib.rcParams = fcm.changeRCParams(
    matplotlib.rcParams,
    color="k",
    tickdir="in",
    mpl=matplotlib,
)

# colormap = matplotlib.cm.get_cmap('hot_ur') #'inferno_r' #'hot_ur'
# colormapBi = matplotlib.cm.get_cmap('coldwhot_u')


def _compute_lims_for_x(radsources: dict):
    if radsources and max([r.energy for r in radsources.values()]) > 40:
        return 2.0, 70.0
    return 2.0, 40.0


def _compute_lims_for_s(radsources: dict):
    return 20.0, 1000.0


def draw_and_save_qlooks(res_cal, path, nthreads=1):
    for quad in res_cal.keys():
        quad_res_cal = res_cal[quad]
        if quad_res_cal.isnull().values.any():
            message = warn_nan_in_sdd_calib(quad)
            logging.warning(message)
            quad_res_cal = quad_res_cal.fillna(0)
        fig, axs = _quicklook(quad_res_cal)
        axs[0].set_title("Calibration quicklook - Quadrant {}".format(quad))
        fig.savefig(path(quad))
        plt.close(fig)
    return


def draw_and_save_slo(res_slo, path, nthreads=1):
    for quad in res_slo.keys():
        quad_res_slo = res_slo[quad]
        if quad_res_slo.isnull().values.any():
            message = warn_nan_in_slo_table(quad)
            logging.warning(message)
            quad_res_slo = quad_res_slo.fillna(0)
        fig, ax = _sloplot(quad_res_slo)
        ax.set_title("Light output - Quadrant {}".format(quad))
        fig.savefig(path(quad))
        plt.close(fig)


def draw_and_save_uncalibrated(xhistograms, shistograms, path, nthreads=1):
    def helper(quad):
        for ch in range(32):
            fig, ax = uncalibrated(
                xhistograms.bins,
                xhistograms.counts[quad][ch],
                shistograms.bins,
                shistograms.counts[quad][ch],
                figsize=(9, 4.5),
            )
            ax.set_title("Uncalibrated plot - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(
        delayed(helper)(quad) for quad in xhistograms.counts.keys()
    )


def draw_and_save_diagns(histograms, res_fit, path, margin=500, nthreads=1):
    def helper(quad):
        for ch in res_fit[quad].index:
            fig, ax = _diagnostics(
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
                margin=margin,
                figsize=(9, 4.5),
                dpi=150,
            )
            ax.set_title("Diagnostic plot - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads, max_nbytes=None)(
        delayed(helper)(quad) for quad in res_fit.keys()
    )


def draw_and_save_channels_xspectra(
    histograms, res_cal, radsources: dict, path, nthreads=1
):
    def helper(quad):
        for ch in res_cal[quad].index:
            enbins = (
                histograms.bins - res_cal[quad].loc[ch]["offset"]
            ) / res_cal[quad].loc[ch]["gain"]
            fig, ax = _spectrum(
                enbins,
                histograms.counts[quad][ch],
                radsources,
                elims=_compute_lims_for_x(radsources),
                figsize=(9, 4.5),
            )
            ax.set_title("Spectra plot X - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(
        delayed(helper)(quad) for quad in res_cal.keys()
    )


def draw_and_save_channels_sspectra(
    histograms, res_cal, res_slo, radsources: dict, path, nthreads=1
):
    def helper(quad):
        for ch in res_slo[quad].index:
            xenbins = (
                histograms.bins - res_cal[quad].loc[ch]["offset"]
            ) / res_cal[quad].loc[ch]["gain"]
            enbins = (
                xenbins / res_slo[quad]["light_out"].loc[ch] / PHOTOEL_PER_KEV
            )

            fig, ax = _spectrum(
                enbins,
                histograms.counts[quad][ch],
                radsources,
                elims=_compute_lims_for_s(radsources),
                figsize=(9, 4.5),
            )
            ax.set_title("Spectra plot S - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(
        delayed(helper)(quad) for quad in res_slo.keys()
    )


def draw_and_save_calibrated_spectra(
    calibrated_events, xradsources: dict, sradsources: dict, xpath, spath
):
    draw_and_save_xspectrum(calibrated_events, xradsources, xpath)
    draw_and_save_sspectrum(calibrated_events, sradsources, spath)
    return True


def draw_and_save_xspectrum(calibrated_events, radsources: dict, path):
    if not calibrated_events.empty:
        xevs = calibrated_events[calibrated_events["EVTYPE"] == "X"]
        xcounts, xbins = np.histogram(
            xevs["ENERGY"], bins=np.arange(2, 40, 0.05)
        )

        fig, ax = _spectrum(
            xbins,
            xcounts,
            radsources,
            elims=_compute_lims_for_x(radsources),
            figsize=(9, 4.5),
            dpi=150,
        )
        ax.set_title("Spectrum X")
        fig.savefig(path)
        plt.close(fig)
        return True


def draw_and_save_sspectrum(calibrated_events, radsources: dict, path):
    if not calibrated_events.empty:
        sevs = calibrated_events[calibrated_events["EVTYPE"] == "S"]
        scounts, sbins = np.histogram(
            sevs["ENERGY"], bins=np.arange(30, 1000, 2)
        )

        fig, ax = _spectrum(
            sbins,
            scounts,
            radsources,
            elims=_compute_lims_for_s(radsources),
            figsize=(9, 4.5),
            dpi=150,
        )
        ax.set_title("Spectrum S")
        fig.savefig(path)
        plt.close(fig)
        return True


def draw_and_save_lins(res_cal, res_fit, radsources: dict, path, nthreads=1):
    def helper(quad):
        for ch in res_cal[quad].index:
            fig, ax = _linearity(
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

    return Parallel(n_jobs=nthreads)(
        delayed(helper)(quad) for quad in res_cal.keys()
    )


def uncalibrated(xbins, xcounts, sbins, scounts, **kwargs):
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.step(xbins[:-1], xcounts, label="X events", where="post")
    ax.fill_between(xbins[:-1], xcounts, step="post", alpha=0.2)
    ax.step(sbins[:-1], scounts, label="S events", where="post")
    ax.fill_between(sbins[:-1], scounts, step="post", alpha=0.2)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Counts")
    ax.set_xlabel("ADU")
    ax.legend()
    return fig, ax


normal = (
    lambda x, amp, sigma, x0: amp
    * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    / (sigma * sqrt(2 * pi))
)


def _diagnostics(
    bins, counts, centers, amps, fwhms, limits, margin=500, **kwargs
):
    low_lims, high_lims = [*zip(*limits)]
    min_xlim, max_xlim = min(low_lims) - margin, max(high_lims) + margin
    start = np.where(bins >= min_xlim)[0][0]
    stop = np.where(bins < max_xlim)[0][-1]
    min_ylim, max_ylim = 0, max(counts[start:stop]) * 1.1

    colors = [plt.cm.tab10(i) for i in range(1, len(limits) + 1)]
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.step(bins[:-1], counts, where="post")
    ax.fill_between(bins[:-1], counts, step="post", alpha=0.4)
    for ctr, amp, fwhm, lims, col in zip(centers, amps, fwhms, limits, colors):
        ax.axvline(ctr, linestyle="dotted")
        ax.axvspan(*lims, color=col, alpha=0.1)
        xs = np.linspace(*lims, 200)
        ax.plot(xs, normal(xs, amp, fwhm / 2.355, ctr), color=col)
    ax.set_xlim((min_xlim, max_xlim))
    ax.set_ylim((min_ylim, max_ylim))
    ax.set_ylabel("Counts")
    ax.set_xlabel("ADU")
    return fig, ax


def _spectrum(enbins, counts, radsources: dict, elims=None, **kwargs):
    radsources_keys = radsources.keys()
    radsources_energies = [l.energy for l in radsources.values()]
    colors = [plt.cm.tab10(i) for i in range(len(radsources))]

    fig, ax = plt.subplots(**kwargs)
    if elims:
        lo, hi = elims
        mask = (enbins[:-1] >= lo) & (enbins[:-1] < hi)
        xs, ys = enbins[:-1][mask], counts[mask]

        ax.set_xlim((lo, hi))
    else:
        xs, ys = enbins[:-1], counts
    ax.step(xs, ys, where="post")
    ax.fill_between(xs, ys, step="post", alpha=0.4)
    for key, value, col in zip(radsources_keys, radsources_energies, colors):
        ax.axvline(value, linestyle="dashed", color=col, label=key)
        ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Counts")
    return fig, ax


def _linearity(
    gain,
    gain_err,
    offset,
    offset_err,
    adcs,
    adcs_err,
    radsources: dict,
    **kwargs
):
    radsources_energies = np.array([l.energy for l in radsources.values()])
    measured_energies_err = np.sqrt(
        (adcs_err**2) * (1 / gain) ** 2
        + (gain_err**2) * ((adcs - offset) / gain**2) ** 2
        + (offset_err**2) * (1 / gain) ** 2
    )
    residual = gain * radsources_energies + offset - adcs
    res_err = np.sqrt(
        (gain_err**2) * (radsources_energies**2)
        + offset_err**2
        + adcs_err**2
    )
    perc_residual = 100 * residual / adcs
    perc_residual_err = 100 * res_err / adcs

    prediction_discrepancy = (adcs - offset) / gain - radsources_energies
    perc_prediction_discrepancy = (
        100 * prediction_discrepancy / radsources_energies
    )
    perc_measured_energies_err = (
        100 * measured_energies_err / radsources_energies
    )

    margin = (radsources_energies[-1] - radsources_energies[0]) / 10
    xs = np.linspace(
        radsources_energies[0] - margin, radsources_energies[-1] + margin, 10
    )

    fig, axs = plt.subplots(
        3, 1, gridspec_kw={"height_ratios": [6, 3, 3]}, sharex=True, **kwargs
    )
    axs[0].errorbar(radsources_energies, adcs, yerr=adcs_err, fmt="o")
    axs[0].plot(xs, gain * xs + offset)
    axs[1].errorbar(
        radsources_energies,
        perc_residual,
        yerr=perc_residual_err,
        fmt="o",
        capsize=5,
    )
    axs[2].errorbar(
        radsources_energies,
        perc_prediction_discrepancy,
        yerr=perc_measured_energies_err,
        fmt="o",
        capsize=5,
    )
    axs[0].set_ylabel("ADC")
    axs[1].set_ylabel("Residual [%]")
    axs[2].set_ylabel("Prediction error [%]")
    axs[2].set_xlabel("Energy [keV]")
    return fig, axs


def _quicklook(calres, **kwargs):
    percentiles = (25, 75)
    gainpercs = np.percentile(calres["gain"], percentiles)
    offsetpercs = np.percentile(calres["offset"], percentiles)

    fig, axs = plt.subplots(2, 1, sharex=True, **kwargs)
    axs[0].errorbar(
        calres.index, calres["gain"],
        yerr=calres["gain_err"],
        fmt="o",
    )
    axs[0].axhspan(
        *gainpercs,
        color="red",
        alpha=0.1,
        label="${}$-${}$ percentiles".format(*percentiles),
    )
    for vg in gainpercs:
        axs[0].axhline(vg, color="r", lw=1)
    axs[0].set_ylabel("Gain")
    axs[0].set_xlim((0, 32))
    axs[0].legend()

    axs[1].errorbar(
        calres.index,
        calres["offset"],
        yerr=calres["offset_err"],
        fmt="o",
    )
    axs[1].axhspan(*offsetpercs, color="red", alpha=0.1)
    for vo in offsetpercs:
        axs[1].axhline(vo, color="r", lw=1)
    axs[1].set_xticks(calres.index)
    axs[1].set_xticklabels(calres.index, rotation=45)
    axs[1].minorticks_off()
    axs[1].set_xlabel("Channel")
    axs[1].set_ylabel("Offset")
    axs[1].set_xlim((0, 32))
    return fig, axs


def _sloplot(res_slo, **kwargs):
    percentiles = (25, 75)
    ypercs = np.percentile(res_slo["light_out"], percentiles)

    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.errorbar(
        res_slo.index,
        res_slo["light_out"],
        yerr=res_slo["light_out_err"],
        fmt="o",
    )
    ax.axhspan(
        *ypercs,
        color="red",
        alpha=0.1,
        label="${}$-${}$ percentiles".format(*percentiles),
    )
    for vg in ypercs:
        ax.axhline(vg, color="r", lw=1)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Light Output [ph/keV]")
    ax.set_xticks(res_slo.index)
    ax.set_xticklabels(res_slo.index, rotation=45, ha="right")
    ax.minorticks_off()
    ax.set_xlim((0, 32))
    ax.legend()
    return fig, ax


# mapplot utilities

quadtext = np.array(
    [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
     [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
     [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
     [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
     [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
     [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
     [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]]
).astype(int)


def _transf(mat, val=-1):
    """
    transform a 12x10 matrix of values into a 24x20 matrix, filling
    empty slots with some value (by default, -1).
    used in map plots.
    """
    m1 = np.dstack((mat, val*np.ones((12, 10)))).reshape((12, 20))
    m2 = np.hstack((m1, val*np.ones((12, 20)))).reshape((24, 20))
    m2 = m2[:-1, :-1]
    return m2


def _chtext(detmap):
    from source.detectors import UNBOND

    chtext = np.zeros((12, 10)).astype(int)
    for i, quad, (tx, ty) in zip(
        [0, 1, 2, 3], ["A", "B", "C", "D"], [(0, 0), (5, 0), (0, 6), (5, 6)]
    ):
        quadmap = np.array(detmap[quad])
        rows, cols = quadmap[
            (quadmap != UNBOND)[:, 0] & (quadmap != UNBOND)[:, 1]
            ].T
        chtext[rows + ty, cols + tx] = np.arange(len(quadmap))[
            (quadmap != UNBOND)[:, 0] & (quadmap != UNBOND)[:, 1]
            ]
    return chtext


def _grid(n, margin, spacing):
    """
    helper function.
    generates 1-D arrays for map plots boundaries.
    given some margin and spacing parameters
    For n=4, margin = 1 and spacing = 3:
    @_@_@_@___@_@_@_@
    """
    assert 0 < margin < 1
    assert spacing > 0
    w = 1 - margin
    op = [(w + margin / 2) * i for i in range(n)]
    cl = [(w - margin / 2) + (w + margin / 2) * i for i in range(n)]
    opcl = np.dstack((op, cl)).reshape(-1)
    axis = np.concatenate((opcl, opcl + n + spacing))
    return axis


def _mapplot(mat, detmap, colorlabel, maskvalue=None, **kwargs):
    xs = _grid(5, margin=0.1, spacing=0.5)
    ys = _grid(6, margin=0.1, spacing=0.3)
    chtext = _chtext(detmap)
    zs = _transf(mat)

    fig, ax = plt.subplots(**kwargs)
    pos = ax.pcolormesh(xs, ys, zs[::-1], vmin=zs[zs > 0].min())
    if maskvalue is not None:
        zm = np.ma.masked_not_equal(zs, 0)
        plt.pcolor(xs, ys, zm[::-1], hatch="///", alpha=0.0)
    wx = xs[2] - xs[1]
    wy = ys[2] - ys[1]
    for i in range(10):
        for j in range(12):
            quad = quadtext[::-1][j, i]
            ax.text(
                (xs[2 * i] + xs[2 * i + 1]) / 2 - wx,
                ys[2 * j] + wy,
                "{}{:02d}".format(
                    ["A", "B", "C", "D"][quad],
                    chtext[::-1][j, i],
                ),
                color="gainsboro",
            )
    ax.set_axis_off()
    fig.colorbar(
        pos,
        label=colorlabel,
        ax=ax,
        aspect=30,
        pad=wy / 10,
        location="bottom",
    )
    return fig, ax


def mapenres(source, en_res, detmap, **kwargs):
    mat = np.zeros((12, 10))

    for i, quad, (tx, ty) in zip(
        [0, 1, 2, 3],
        ["A", "B", "C", "D"],
        [(0, 0), (5, 0), (0, 6), (5, 6)],
    ):
        if quad in en_res.keys():
            quadmap = np.array(detmap[quad])
            channels = en_res[quad][source].index
            chns_indeces = quadmap[channels]
            values = en_res[quad][source]["resolution"].values

            rows, cols = chns_indeces.T
            mat[rows + ty, cols + tx] = values

    fig, ax = _mapplot(mat, detmap, colorlabel="Energy resolution [keV]", maskvalue=0, **kwargs)
    return fig, ax


def mapcounts(counts, detmap, **kwargs):
    mat = np.zeros((12, 10))

    for i, quad, (tx, ty) in zip(
        [0, 1, 2, 3],
        ["A", "B", "C", "D"],
        [(0, 0), (5, 0), (0, 6), (5, 6)],
    ):
        if quad in counts.keys():
            quadmap = np.array(detmap[quad])
            channels = counts[quad].index
            chns_indeces = quadmap[channels]
            values = counts[quad]['counts'].values

            rows, cols = chns_indeces.T
            mat[rows + ty, cols + tx] = values

    fig, ax = _mapplot(mat, detmap, colorlabel="Counts", maskvalue=0, **kwargs)
    return fig, ax