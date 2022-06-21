import numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn')
from joblib import Parallel, delayed


def draw_and_save_slo(res_slo, path, nthreads=1):
    def helper(asic):
        fig, ax = sloplot(res_slo[asic])
        ax.set_title("Light output - Quadrant {}".format(asic))
        fig.savefig(path(asic))
        plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(asic) for asic in res_slo.keys())


def draw_and_save_uncalibrated(xbins, xhists, sbins, shists, path, nthreads=1):
    def helper(asic):
        for ch in range(32):
            xcounts = xhists[asic][ch]
            scounts = shists[asic][ch]
            fig, ax = uncalibrated(xbins, xcounts, sbins, scounts,
                                   figsize=(9, 4.5))
            ax.set_title("Uncalibrated plot - CH{:02d}Q{}".format(ch, asic))
            fig.savefig(path(asic, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(asic) for asic in xhists.keys())


def draw_and_save_diagns(bins, hists, res_fit, path, nthreads=1):
    def helper(asic):
        for ch in res_fit[asic].index:
            fig, ax = diagnostics(bins,
                                  hists[asic][ch],
                                  res_fit[asic].loc[ch].loc[:, 'center'],
                                  res_fit[asic].loc[ch].loc[:, ['lim_low', 'lim_high']].values.reshape(2, -1).T,
                                  figsize=(9, 4.5))
            ax.set_title("Diagnostic plot - CH{:02d}Q{}".format(ch, asic))
            fig.savefig(path(asic, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(asic) for asic in res_fit.keys())


def draw_and_save_channels_xspectra(bins, hists, res_cal, lines, path, nthreads=1):
    def helper(asic):
        for ch in res_cal[asic].index:
            enbins = (bins - res_cal[asic].loc[ch]['offset']) / res_cal[asic].loc[ch]['gain']
            fig, ax = spectrum(enbins,
                               hists[asic][ch],
                               lines,
                               elims=(2.0, 40.0),
                               figsize=(9, 4.5))
            ax.set_title("Spectra plot - CH{:02d}Q{}".format(ch, asic))
            fig.savefig(path(asic, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(asic) for asic in res_cal.keys())


def draw_and_save_xspectrum(calibrated_events, lines, path):
    if not calibrated_events.empty:
        xevs = calibrated_events[calibrated_events['EVTYPE'] == 'X']
        xcounts, xbins = np.histogram(xevs['ENERGY'], bins=np.arange(2, 40, 0.05))

        fig, ax = spectrum(xbins,
                           xcounts,
                           lines,
                           elims=(2.0, 40.0),
                           figsize=(9, 4.5))
        ax.set_title("Spectrum X")
        fig.savefig(path)
        plt.close(fig)
        return True


def draw_and_save_sspectrum(calibrated_events, lines, path):
    if not calibrated_events.empty:
        sevs = calibrated_events[calibrated_events['EVTYPE'] == 'S']
        scounts, sbins = np.histogram(sevs['ENERGY'], bins=np.arange(30, 1000, 2))

        fig, ax = spectrum(sbins,
                           scounts,
                           lines,
                           elims=(30, 1000),
                           figsize=(9, 4.5))
        ax.set_title("Spectrum S")
        fig.savefig(path)
        plt.close(fig)
        return True


def draw_and_save_lins(res_cal, res_fit, lines, path, nthreads=1):
    def helper(asic):
        for ch in res_cal[asic].index:
            fig, ax = linearity(*res_cal[asic].loc[ch][['gain', 'gain_err', 'offset', 'offset_err']],
                                res_fit[asic].loc[ch].loc[:, 'center'],
                                res_fit[asic].loc[ch].loc[:, 'center_err'],
                                lines)
            ax[0].set_title("Linearity plot - CH{:02d}Q{}".format(ch, asic))
            fig.savefig(path(asic, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(asic) for asic in res_cal.keys())


def draw_and_save_qlooks(res_cal, path, nthreads=1):
    def helper(asic):
        fig, axs = quicklook(res_cal[asic])
        axs[0].set_title("Calibration quicklook - Quadrant {}".format(asic))
        fig.savefig(path(asic))
        plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(asic) for asic in res_cal.keys())


def uncalibrated(xbins, xcounts, sbins, scounts, **kwargs):
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.step(xbins[:-1], xcounts, label='X events', where='post')
    ax.fill_between(xbins[:-1], xcounts, step="post", alpha=0.2)
    ax.step(sbins[:-1], scounts, color='tomato', label='S events', where='post')
    ax.fill_between(sbins[:-1], scounts, step="post", alpha=0.2, color='tomato')
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


def spectrum(enbins, counts, lines: dict={}, elims=None, **kwargs):
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
    axs[0].set_xlim((0,32))
    axs[0].legend()

    axs[1].errorbar(calres.index, calres['offset'], yerr=calres['offset_err'], fmt='o')
    axs[1].axhspan(*offsetpercs, color='red', alpha=0.1)
    for vo in offsetpercs:
        axs[1].axhline(vo, color='r', lw=1)
    axs[1].set_xticks(calres.index)
    axs[1].set_xlabel("Channel")
    axs[1].set_ylabel("Offset")
    axs[1].set_xlim((0,32))
    return fig, axs


def sloplot(res_slo, **kwargs):
    x = res_slo.index
    y = res_slo.groupby(axis=1,level=-1).mean()['light_out']
    yerr = res_slo.groupby(axis=1,level=-1).mean()['light_out_err']
    #y = res_slo['light_out'].T.mean()
    #yerr = res_slo['light_out_err'].T.mean()
    ypercs = np.percentile(y, [30, 70])

    fig, ax = plt.subplots(1,1, **kwargs)
    ax.errorbar(x, y, yerr = yerr, fmt = 'o')
    ax.axhspan(*ypercs, color='red', alpha=0.1, label='$30$-$70$ percentiles')
    for vg in ypercs:
        ax.axhline(vg, color='r', lw=1)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Light Output [ph/keV]")
    ax.set_xticks(x)
    ax.set_xlim((0,32))
    ax.legend()
    return fig, ax
