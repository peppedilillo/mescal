import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')


def diagnostics(bins, counts, centers, limits, **kwargs):
    colors = [plt.cm.tab10(i) for i in range(len(limits))]

    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.step(bins[:-1], counts)
    ax.fill_between(bins[:-1], counts, step="pre", alpha=0.4)
    for ctr, lims, col in zip(centers, limits, colors):
        ax.axvline(ctr, linestyle='dotted')
        ax.axvspan(*lims, color=col, alpha=0.1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Counts")
    ax.set_xlabel("ADU")
    return fig, ax


def spectrum(enbins, counts, lines: dict, elims=None, **kwargs):
    colors = [plt.cm.tab10(i) for i in range(len(lines))]

    fig, ax = plt.subplots(**kwargs)
    ax.step(enbins[:-1], counts)
    ax.fill_between(enbins[:-1], counts, step="pre", alpha=0.4)
    for (lines_keys, lines_values), col in zip(lines.items(), colors):
        ax.axvline(lines_values, linestyle="dashed", color=col, label=lines_keys)
    if elims:
        ax.set_xlim(*elims)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Counts')
    ax.legend(loc="upper right")
    return fig, ax


def linearity(gain, gain_err, offset, offset_err, adcs, adcs_err, lines: dict, **kwargs):
    _, ls = zip(*lines.items())
    ls = np.array(ls)
    residual = gain * ls + offset - adcs
    res_err = np.sqrt((gain_err ** 2) * (ls ** 2) + offset_err ** 2 + adcs_err ** 2)
    perc_res = 100 * residual / adcs
    perc_res_err = 100 * res_err / adcs

    margin = (ls[-1] - ls[0]) / 10
    xs = np.linspace(ls[0] - margin, ls[-1] + margin, 10)

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, tight_layout=True, **kwargs)
    axs[0].errorbar(ls, adcs, yerr=adcs_err, fmt='o')
    axs[0].plot(xs, gain * xs + offset)
    axs[1].errorbar(ls, perc_res, yerr=perc_res_err, fmt='o', capsize=5)

    axs[0].set_ylabel("ADU")
    axs[1].set_ylabel("Residuals [%]")
    axs[1].set_xlabel("Energy [keV]")
    return fig, axs


def quicklook(calres, **kwargs):
    gainpercs = np.percentile(calres['gain'], [30, 70])
    offsetpercs = np.percentile(calres['offset'], [30, 70])

    fig, axs = plt.subplots(2, 1, sharex=True, **kwargs)
    axs[0].errorbar(calres.index, calres['gain'], yerr=calres['gain_err'], fmt='o')
    axs[0].axhspan(*gainpercs, color='red', alpha=0.1, label='$30$-$70$ percentiles')
    for vg in gainpercs:
        axs[0].axhline(vg, color='r', lw=1)
    axs[0].set_ylabel("Gain")
    axs[0].set_title("Calibration quicklook")
    axs[0].legend()

    axs[1].errorbar(calres.index, calres['offset'], yerr=calres['offset_err'], fmt='o')
    axs[1].axhspan(*offsetpercs, color='red', alpha=0.1)
    for vo in offsetpercs:
        axs[1].axhline(vo, color='r', lw=1)
    axs[1].set_xticks(calres.index)
    axs[1].set_xlabel("Channel")
    axs[1].set_ylabel("Offset")
    return fig, axs
