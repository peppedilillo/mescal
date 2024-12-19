from math import pi
from math import sqrt

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

import source.fcparams as fcm

matplotlib.rcParams = fcm.changeRCParams(
    matplotlib.rcParams,
    color="k",
    tickdir="in",
    mpl=matplotlib,
)


def _compute_lims_for_x(radsources: dict):
    if radsources and max([r.energy for r in radsources.values()]) > 40:
        return 2.0, 70.0
    return 2.0, 40.0


def _compute_lims_for_s():
    return 50.0, 1000.0


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


def diagnostics(
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


def spectrum_x(enbins, counts, radsources: dict, **kwargs):
    fig, ax = _spectrum(
        enbins,
        counts,
        radsources,
        elims=_compute_lims_for_x(radsources),
        **kwargs,
    )
    return fig, ax


def spectrum_s(enbins, counts, radsources: dict, **kwargs):
    fig, ax = _spectrum(
        enbins,
        counts,
        radsources,
        elims=_compute_lims_for_s(),
        **kwargs,
    )
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


def spectrum_xs(
    calibrated_events,
    xradsources: dict,
    sradsources: dict,
    xlims=None,
    slims=None,
    **kwargs,
):
    xevs = calibrated_events[calibrated_events["EVTYPE"] == "X"]
    xcounts, xbins = np.histogram(
        xevs["ENERGY"],
        bins=np.arange(*_compute_lims_for_x(xradsources), 0.05),
    )
    sevs = calibrated_events[calibrated_events["EVTYPE"] == "S"]
    scounts, sbins = np.histogram(
        sevs["ENERGY"],
        bins=np.arange(*_compute_lims_for_s(), 0.5),
    )

    fig, axs = plt.subplots(2, 1, **kwargs)
    for ax, radsources, (enbins, counts), elims, color in zip(
        axs,
        (xradsources, sradsources),
        ((xbins, xcounts), (sbins, scounts)),
        (xlims, slims),
        plt.rcParams["axes.prop_cycle"].by_key()["color"][:2],
    ):
        radsources_keys = radsources.keys()
        radsources_energies = [l.energy for l in radsources.values()]
        if elims:
            lo, hi = elims
            mask = (enbins[:-1] >= lo) & (enbins[:-1] < hi)
            xs, ys = enbins[:-1][mask], counts[mask]
        else:
            lo, hi = enbins[0], enbins[-1]
            xs, ys = enbins[:-1], counts

        ax.step(xs, ys, where="post", color=color)
        ax.fill_between(xs, ys, step="post", alpha=0.4, color=color)
        for key, value in zip(radsources_keys, radsources_energies):
            ax.axvline(value, linestyle="dashed", label=key)
            ax.legend(loc="upper right")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=lo, right=hi)
    axs[0].set_title("Calibrated spectra")
    fig.supylabel("Counts")
    fig.supxlabel("Energy [keV]")
    return fig, axs


def linearity(
    gain,
    gain_err,
    offset,
    offset_err,
    adcs,
    adcs_err,
    radsources: dict,
    **kwargs,
):
    radsources_energies = np.array([l.energy for l in radsources.values()])
    measured_energies_err = np.sqrt(
        (adcs_err**2) * (1 / gain) ** 2
        + (gain_err**2) * ((adcs - offset) / gain**2) ** 2
        + (offset_err**2) * (1 / gain) ** 2
    )
    residual = gain * radsources_energies + offset - adcs
    res_err = np.sqrt(
        (gain_err**2) * (radsources_energies**2) + offset_err**2 + adcs_err**2
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


def quicklook(calres, **kwargs):
    percentiles = (25, 75)
    gainpercs = np.percentile(calres["gain"], percentiles)
    offsetpercs = np.percentile(calres["offset"], percentiles)

    fig, axs = plt.subplots(2, 1, sharex=True, **kwargs)
    axs[0].errorbar(
        calres.index,
        calres["gain"],
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


def lightout(res_slo, **kwargs):
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


def histogram(counts, bins, **kwargs):
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.step(bins, counts)
    ax.set_ylim(bottom=0)
    return fig, ax


# mapplot utilities

_quadtext = np.array(
    [
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
    ]
).astype(int)

_quadcorners = [(0, 0), (0, 6), (5, 0), (5, 6)]


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
    """
    TODO: this is absolutely obscure and sucks. also its importing UNBOND which
          should be made private to detectors.py.
    """
    from source.detectors import UNBOND

    chtext = np.zeros((12, 10)).astype(int)
    for i, quad, (tx, ty) in zip(
        [0, 1, 2, 3],
        ["A", "B", "C", "D"],
        _quadcorners,
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


def _mapplot(mat, detmap, colorlabel, cmap="hot_ur", maskvalue=None, **kwargs):
    xs = _grid(5, margin=0.1, spacing=0.5)
    ys = _grid(6, margin=0.1, spacing=0.3)
    chtext = _chtext(detmap)
    zs = _transf(mat)

    fig, ax = plt.subplots(**kwargs)
    pos = ax.pcolormesh(
        xs,
        ys,
        zs[::-1],
        vmin=zs[zs > 0].min(),
        cmap=cmap,
    )
    if maskvalue is not None:
        zm = np.ma.masked_not_equal(zs, 0)
        plt.pcolor(xs, ys, zm[::-1], hatch="///", alpha=0.0)
    wx = xs[2] - xs[1]
    wy = ys[2] - ys[1]
    for i in range(10):
        for j in range(12):
            quad = _quadtext[::-1][j, i]
            text = ax.text(
                (xs[2 * i] + xs[2 * i + 1]) / 2 - wx,
                ys[2 * j] + wy,
                "{}{:02d}".format(
                    ["A", "B", "C", "D"][quad],
                    chtext[::-1][j, i],
                ),
                color="white",
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1, foreground="black"),
                    path_effects.Normal(),
                ]
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


def mapenres(
    source: str,
    en_res,
    detmap,
    cmap="cold_ur",
):
    mat = np.zeros((12, 10))

    for i, quad, (tx, ty) in zip(
        [0, 1, 2, 3],
        ["A", "B", "C", "D"],
        _quadcorners,
    ):
        if quad in en_res.keys():
            quadmap = np.array(detmap[quad])
            channels = en_res[quad][source].index
            chns_indeces = quadmap[channels]
            values = en_res[quad][source]["resolution"].values

            rows, cols = chns_indeces.T
            mat[rows + ty, cols + tx] = values

    fig, ax = _mapplot(
        mat,
        detmap,
        cmap=cmap,
        colorlabel="Energy resolution [keV]",
        maskvalue=0,
        figsize=(8, 8),
    )
    ax.set_title("{} energy resolution".format(source))
    return fig, ax


def mapcounts(counts, detmap, cmap="hot_ur", title=None):
    mat = np.zeros((12, 10))

    for i, quad, (tx, ty) in zip(
        [0, 1, 2, 3],
        ["A", "B", "C", "D"],
        _quadcorners,
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

    fig, ax = _mapplot(
        mat,
        detmap,
        cmap=cmap,
        colorlabel="Counts",
        maskvalue=0,
        figsize=(8, 8),
    )
    if title is not None:
        ax.set_title(title)
    return fig, ax
