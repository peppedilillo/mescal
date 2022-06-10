import matplotlib.pyplot as plt;

plt.style.use('seaborn')
from joblib import Parallel, delayed
import numpy as np


def draw_and_save_uncalibrated(bins, chndic, data, saveto, ncore=1):
    def helper(asic):
        for ch in chndic[asic]:
            fig, ax = uncalibrated(bins,
                                   data[(data['QUADID'] == asic) & (data['CHN'] == ch)],
                                   figsize=(9, 4.5))
            ax.set_title("Uncalibrated plot - CH{:02d}Q{}".format(ch, asic))
            fig.savefig(saveto(asic, ch))
            plt.close(fig)

    Parallel(n_jobs=ncore)(delayed(helper)(asic) for asic in chndic.keys())
    return True


def draw_and_save_diagns(bins, hists, res_fit, saveto, ncore=1):
    def helper(asic):
        for ch in res_fit[asic].index:
            fig, ax = diagnostics(bins,
                                  hists[asic][ch],
                                  res_fit[asic].loc[ch]['center'],
                                  res_fit[asic].loc[ch][['lim_low', 'lim_high']].unstack(level=0).values,
                                  figsize=(9, 4.5))
            ax.set_title("Diagnostic plot - CH{:02d}Q{}".format(ch, asic))
            fig.savefig(saveto(asic, ch))
            plt.close(fig)

    Parallel(n_jobs=ncore)(delayed(helper)(asic) for asic in res_fit.keys())
    return True


def draw_and_save_xspectra(bins, hists, res_cal, lines, saveto, ncore=1):
    def helper(asic):
        for ch in res_cal[asic].index:
            enbins = (bins - res_cal[asic].loc[ch]['offset']) / res_cal[asic].loc[ch]['gain']
            fig, ax = spectrum(enbins,
                               hists[asic][ch],
                               lines,
                               elims=(2.0, 40.0),
                               figsize=(9, 4.5))
            ax.set_title("Spectra plot - CH{:02d}Q{}".format(ch, asic))
            fig.savefig(saveto(asic, ch))
            plt.close(fig)

    Parallel(n_jobs=ncore)(delayed(helper)(asic) for asic in res_cal.keys())
    return True


def draw_and_save_lins(res_cal, res_fit, lines, saveto, ncore=1):
    def helper(asic):
        for ch in res_cal[asic].index:
            fig, ax = linearity(*res_cal[asic].loc[ch][['gain', 'gain_err', 'offset', 'offset_err']],
                                res_fit[asic].loc[ch][['center']].values,
                                res_fit[asic].loc[ch][['center_err']].values,
                                lines)
            ax[0].set_title("Linearity plot - CH{:02d}Q{}".format(ch, asic))
            fig.savefig(saveto(asic, ch))
            plt.close(fig)

    Parallel(n_jobs=ncore)(delayed(helper)(asic) for asic in res_cal.keys())
    return True


def draw_and_save_qlooks(res_cal, saveto, ncore=1):
    def helper(asic):
        fig, axs = quicklook(res_cal[asic])
        axs[0].set_title("Calibration quicklook - Quadrant {}".format(asic))
        fig.savefig(saveto(asic))
        plt.close(fig)

    Parallel(n_jobs=ncore)(delayed(helper)(asic) for asic in res_cal.keys())
    return True


def uncalibrated(bins, chdata, **kwargs):
    counts_x, _ = np.histogram(chdata[chdata['EVTYPE'] == 'X']['ADC'], bins=bins)
    counts_s, _ = np.histogram(chdata[chdata['EVTYPE'] == 'S']['ADC'], bins=bins)

    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.step(bins[:-1], counts_x, label='X events', where='post')
    ax.fill_between(bins[:-1], counts_x, step="post", alpha=0.2)
    ax.step(bins[:-1], counts_s, color='tomato', label='S events', where='post')
    ax.fill_between(bins[:-1], counts_s, step="post", alpha=0.2, color='tomato')
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Counts")
    ax.set_xlabel("ADU")
    ax.legend()
    return fig, ax


def diagnostics(bins, counts, centers, limits, **kwargs):
    colors = [plt.cm.tab10(i) for i in range(len(limits))]

    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.step(bins[:-1], counts, where='post')
    ax.fill_between(bins[:-1], counts, step="post", alpha=0.4)
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
    if elims:
        lo, hi = elims
        mask = (enbins[:-1] >= lo) & (enbins[:-1] < hi)
        xs, ys = enbins[:-1][mask], counts[mask]

        ax.set_xlim((lo, hi))
    else:
        xs, ys = enbins[:-1], counts
    ax.step(xs, ys, where='post')
    ax.fill_between(xs, ys, step="post", alpha=0.4)
    for (lines_keys, lines_values), col in zip(lines.items(), colors):
        ax.axvline(lines_values, linestyle="dashed", color=col, label=lines_keys)
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
    axs[0].legend()

    axs[1].errorbar(calres.index, calres['offset'], yerr=calres['offset_err'], fmt='o')
    axs[1].axhspan(*offsetpercs, color='red', alpha=0.1)
    for vo in offsetpercs:
        axs[1].axhline(vo, color='r', lw=1)
    axs[1].set_xticks(calres.index)
    axs[1].set_xlabel("Channel")
    axs[1].set_ylabel("Offset")
    return fig, axs
