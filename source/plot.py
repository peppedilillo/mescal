import logging
from math import pi, sqrt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

import source.fcparams as fcm
from source.constants import PHOTOEL_PER_KEV
from source.errors import warn_nan_in_sdd_calib, warn_nan_in_slo_table

matplotlib.rcParams = fcm.changeRCParams(
    matplotlib.rcParams, color="k", tickdir="in", mpl=matplotlib,
)


def _compute_lims_for_x(radsources: dict):
    if radsources and max([r.energy for r in radsources.values()]) > 40:
        return 2.0, 70.0
    return 2.0, 40.0


def _compute_lims_for_s(radsources: dict):
    return 20.0, 1000.0


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
    * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    / (sigma * sqrt(2 * pi))
)


def diagnostics(bins, counts, centers, amps, fwhms, limits, margin=500, **kwargs):
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


def spectrum(enbins, counts, radsources: dict, elims=None, **kwargs):
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


def gainoffset(sdd_calibration, **kwargs):
    percentiles = (25, 75)
    gainpercs = np.percentile(sdd_calibration["gain"], percentiles)
    offsetpercs = np.percentile(sdd_calibration["offset"], percentiles)
    channels = sdd_calibration.index
    gains = sdd_calibration["gain"]
    gains_err = sdd_calibration["gain_err"]
    offsets = sdd_calibration["offset"]
    ossets_err = sdd_calibration["offset_err"]

    fig, axs = plt.subplots(2, 1, sharex=True, **kwargs)
    axs[0].errorbar(channels, gains, yerr=gains_err, fmt="o",)
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

    axs[1].errorbar(channels, offsets, yerr=ossets_err, fmt="o")
    axs[1].axhspan(*offsetpercs, color="red", alpha=0.1)
    for vo in offsetpercs:
        axs[1].axhline(vo, color="r", lw=1)
    axs[1].set_xticks(channels)
    axs[1].set_xticklabels(channels, rotation=45)
    axs[1].minorticks_off()
    axs[1].set_xlabel("Channel")
    axs[1].set_ylabel("Offset")
    axs[1].set_xlim((0, 32))
    return fig, axs


def lightout(effective_lightout, **kwargs):
    percentiles = (25, 75)
    ypercs = np.percentile(effective_lightout["light_out"], percentiles)

    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.errorbar(
        effective_lightout.index,
        effective_lightout["light_out"],
        yerr=effective_lightout["light_out_err"],
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
    ax.set_xticks(effective_lightout.index)
    ax.set_xticklabels(effective_lightout.index, rotation=45, ha="right")
    ax.minorticks_off()
    ax.set_xlim((0, 32))
    ax.legend()
    return fig, ax


# detector map plot utilities
quadtext = np.array(
    [
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
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
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    ]
).astype(int)


def _transf(mat, val=-1):
    """
    transform a 12x10 matrix of values into a 24x20 matrix, filling
    empty slots with some value (by default, -1).
    used in map plots.
    """
    m1 = np.dstack((mat, val * np.ones((12, 10)))).reshape((12, 20))
    m2 = np.hstack((m1, val * np.ones((12, 20)))).reshape((24, 20))
    m2 = m2[:-1, :-1]
    return m2


def _chtext(detmap):
    from source.detectors import UNBOND

    chtext = np.zeros((12, 10)).astype(int)
    for i, quad, (tx, ty) in zip(
        [0, 1, 2, 3], ["A", "B", "C", "D"], [(0, 0), (5, 0), (0, 6), (5, 6)]
    ):
        quadmap = np.array(detmap[quad])
        rows, cols = quadmap[(quadmap != UNBOND)[:, 0] & (quadmap != UNBOND)[:, 1]].T
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
                "{}{:02d}".format(["A", "B", "C", "D"][quad], chtext[::-1][j, i],),
                color="gainsboro",
            )
    ax.set_axis_off()
    fig.colorbar(
        pos, label=colorlabel, ax=ax, aspect=30, pad=wy / 10, location="bottom",
    )
    return fig, ax


def mapenres(source: str, energy_resolution, detmap):
    mat = np.zeros((12, 10))

    for i, quad, (tx, ty) in zip(
        [0, 1, 2, 3], ["A", "B", "C", "D"], [(0, 0), (5, 0), (0, 6), (5, 6)],
    ):
        if quad in energy_resolution.keys():
            quadmap = np.array(detmap[quad])
            channels = energy_resolution[quad][source].index
            chns_indeces = quadmap[channels]
            values = energy_resolution[quad][source]["resolution"].values

            rows, cols = chns_indeces.T
            mat[rows + ty, cols + tx] = values

    fig, ax = _mapplot(
        mat, detmap, colorlabel="Energy resolution [keV]", maskvalue=0, figsize=(8, 8),
    )
    ax.set_title("{} energy resolution".format(source))
    return fig, ax


def maplightout(effective_lightout, detmap):
    mat = np.zeros((12, 10))

    for i, quad, (tx, ty) in zip(
        [0, 1, 2, 3], ["A", "B", "C", "D"], [(0, 0), (5, 0), (0, 6), (5, 6)],
    ):
        if quad in effective_lightout.keys():
            quadmap = np.array(detmap[quad])
            channels = effective_lightout[quad].index
            chns_indeces = quadmap[channels]
            values = effective_lightout[quad]["light_out"].values

            rows, cols = chns_indeces.T
            mat[rows + ty, cols + tx] = values

    fig, ax = _mapplot(
        mat, detmap, colorlabel="Light output (ph/keV)", maskvalue=0, figsize=(8, 8),
    )
    ax.set_title("Light output")
    return fig, ax


def mapcounts(counts, detmap):
    mat = np.zeros((12, 10))

    for i, quad, (tx, ty) in zip(
        [0, 1, 2, 3], ["A", "B", "C", "D"], [(0, 0), (5, 0), (0, 6), (5, 6)],
    ):
        if quad in counts.keys():
            quadmap = np.array(detmap[quad])
            channels = counts[quad].index
            chns_indeces = quadmap[channels]
            values = counts[quad]["counts"].values
            # next line protects against noise from unbound channels
            mask = (quadmap[channels] >= 0).all(axis=1)
            rows, cols = chns_indeces[mask].T
            mat[rows + ty, cols + tx] = values[mask]

    fig, ax = _mapplot(mat, detmap, colorlabel="Counts", maskvalue=0, figsize=(8, 8),)
    ax.set_title("Per-channel counts (pixel events)")
    return fig, ax


# next functions prepare and ship plots via export
def draw_and_save_gainoffset(sdd_calibration, path, nthreads=1):
    for quad in sdd_calibration.keys():
        quad_res_cal = sdd_calibration[quad]
        if quad_res_cal.isnull().values.any():
            message = warn_nan_in_sdd_calib(quad)
            logging.warning(message)
            quad_res_cal = quad_res_cal.fillna(0)
        fig, axs = gainoffset(quad_res_cal)
        axs[0].set_title("Calibration quicklook - Quadrant {}".format(quad))
        fig.savefig(path(quad))
        plt.close(fig)
    return


def draw_and_save_lightout(effective_lightout, path):
    for quad in effective_lightout.keys():
        quad_res_slo = effective_lightout[quad]
        if quad_res_slo.isnull().values.any():
            message = warn_nan_in_slo_table(quad)
            logging.warning(message)
            quad_res_slo = quad_res_slo.fillna(0)
        fig, ax = lightout(quad_res_slo)
        ax.set_title("Light output - Quadrant {}".format(quad))
        fig.savefig(path(quad))
        plt.close(fig)
    return


def draw_and_save_uncalibrated(xhistograms, shistograms, path, nthreads=1):
    def helper(quad):
        for ch in range(32):
            fig, ax = uncalibrated(
                xhistograms.bins,
                xhistograms.counts[quad][ch],
                shistograms.bins,
                shistograms.counts[quad][ch],
                figsize=(8, 4.5),
            )
            ax.set_title("Uncalibrated plot - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(
        delayed(helper)(quad) for quad in xhistograms.counts.keys()
    )


def draw_and_save_diagns(histograms, fit_results, path, margin=500, nthreads=1):
    def helper(quad):
        for ch in fit_results[quad].index:
            fig, ax = diagnostics(
                histograms.bins,
                histograms.counts[quad][ch],
                fit_results[quad].loc[ch].loc[:, "center"],
                fit_results[quad].loc[ch].loc[:, "amp"],
                fit_results[quad].loc[ch].loc[:, "fwhm"],
                fit_results[quad]
                .loc[ch]
                .loc[:, ["lim_low", "lim_high"]]
                .values.reshape(2, -1)
                .T,
                margin=margin,
                figsize=(8, 4.5),
            )
            ax.set_title("Diagnostic plot - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads, max_nbytes=None)(
        delayed(helper)(quad) for quad in fit_results.keys()
    )


def draw_and_save_channels_xspectra(
    histograms, sdd_calibration, radsources: dict, path, nthreads=1
):
    def helper(quad):
        for ch in sdd_calibration[quad].index:
            offsets = sdd_calibration[quad].loc[ch]["offset"]
            gains = sdd_calibration[quad].loc[ch]["gain"]

            enbins = (histograms.bins - offsets) / gains
            fig, ax = spectrum(
                enbins,
                histograms.counts[quad][ch],
                radsources,
                elims=_compute_lims_for_x(radsources),
                figsize=(8, 4.5),
            )
            ax.set_title("Spectra plot X - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in sdd_calibration.keys())


def draw_and_save_channels_sspectra(
    histograms, sdd_calibration, effective_lightout, radsources: dict, path, nthreads=1
):
    def helper(quad):
        for ch in effective_lightout[quad].index:
            offsets = sdd_calibration[quad].loc[ch]["offset"]
            gains = sdd_calibration[quad].loc[ch]["gain"]
            light_outs = effective_lightout[quad]["light_out"].loc[ch]

            xenbins = (histograms.bins - offsets) / gains
            enbins = xenbins / light_outs / PHOTOEL_PER_KEV

            fig, ax = spectrum(
                enbins,
                histograms.counts[quad][ch],
                radsources,
                elims=_compute_lims_for_s(radsources),
                figsize=(8, 4.5),
            )
            ax.set_title("Spectra plot S - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in effective_lightout.keys())


def draw_and_save_calibrated_spectra(
    calibrated_events, xradsources: dict, sradsources: dict, xpath, spath
):
    draw_and_save_xspectrum(calibrated_events, xradsources, xpath)
    draw_and_save_sspectrum(calibrated_events, sradsources, spath)
    return True


def draw_and_save_xspectrum(calibrated_events, radsources: dict, path):
    xevs = calibrated_events[calibrated_events["EVTYPE"] == "X"]
    xcounts, xbins = np.histogram(xevs["ENERGY"], bins=np.arange(2, 40, 0.05))

    fig, ax = spectrum(
        xbins,
        xcounts,
        radsources,
        elims=_compute_lims_for_x(radsources),
        figsize=(8, 4.5),
    )
    ax.set_title("Spectrum X")
    fig.savefig(path)
    plt.close(fig)
    return True


def draw_and_save_sspectrum(calibrated_events, radsources: dict, path):
    sevs = calibrated_events[calibrated_events["EVTYPE"] == "S"]
    scounts, sbins = np.histogram(sevs["ENERGY"], bins=np.arange(30, 1000, 2))

    fig, ax = spectrum(
        sbins,
        scounts,
        radsources,
        elims=_compute_lims_for_s(radsources),
        figsize=(8, 4.5),
    )
    ax.set_title("Spectrum S")
    fig.savefig(path)
    plt.close(fig)
    return True


def draw_and_save_maplightout(effective_lightout, detmap, path):
    fig, ax = maplightout(effective_lightout, detmap)
    fig.savefig(path)
    plt.close(fig)
    return True


def draw_and_save_mapenres(source, energy_resolution, detmap, path):
    fig, ax = mapenres(source, energy_resolution, detmap)
    fig.savefig(path)
    plt.close(fig)
    return True


def draw_and_save_mapcounts(counts, detmap, path):
    fig, ax = mapcounts(counts, detmap)
    fig.savefig(path)
    plt.close(fig)
    return True