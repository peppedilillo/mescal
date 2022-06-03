import matplotlib.pyplot as plt
import numpy as np


def diagnostics(bins, counts, centers, limits, **kwargs):
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.step(bins[:-1], counts)
    for c, l in zip(centers, limits):
        ax.axvline(c, color='red', linestyle='dotted')
        ax.axvspan(*l, color='r', alpha=0.1)
    ax.set_ylabel("Counts")
    ax.set_xlabel("ADU")
    return fig, ax


def spectrum(enbins, counts, lines: dict, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    ax.step(enbins[:-1], counts, color='red', linewidth=0.8)
    for lines_keys, lines_values in lines.items():
        ax.axvline(lines_values, linestyle="dashed", color='green', label=lines_keys, linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('N')
    return fig, ax


def linearity(gain, gain_err, offset, offset_err, adcs, adcs_err, lines, **kwargs):
    """
    TODO: is error propagation ok?
    """
    _, ls = zip(*lines.items())
    margin = (ls[-1] - ls[0]) / 10
    xs = np.linspace(ls[0] - margin, ls[-1] + margin, 10)
    residual = gain * ls + offset - adcs
    res_err = np.sqrt((gain_err ** 2) * (ls ** 2) + offset_err ** 2 + adcs_err ** 2)
    perc_res = 100 * (gain * ls + offset - adcs) / adcs
    perc_res_err = 100 * residual / adcs * np.sqrt(res_err ** 2 / residual ** 2 + adcs_err ** 2 / adcs ** 2
                                                   + 2 * adcs_err ** 2 / (adcs * residual))

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, tight_layout=True, **kwargs)
    axs[0].errorbar(ls, adcs, yerr=adcs_err, fmt='o')
    axs[0].plot(xs, gain * xs + offset, color='red')
    axs[1].errorbar(ls, perc_res, yerr=perc_res_err, fmt='o', capsize=5)

    axs[0].set_title("Linearity Plot")
    axs[0].set_ylabel("ADU")
    axs[1].set_ylabel("Residuals [%]")
    axs[1].set_xlabel("Energy [keV]")
    return fig, axs
