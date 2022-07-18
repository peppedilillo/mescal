from math import sqrt
from math import pi

import numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn')
from joblib import Parallel
from joblib import delayed

from source.spectra import PHT_KEV


def _compute_lims_for_x(radsources: dict):
    return 2., 40.


def _compute_lims_for_s(radsources: dict):
    return 20., 1000.


def draw_and_save_slo(res_slo, path, nthreads=1):
    def helper(quad):
        fig, ax = _sloplot(res_slo[quad])
        ax.set_title("Light output - Quadrant {}".format(quad))
        fig.savefig(path(quad))
        plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in res_slo.keys())


def draw_and_save_uncalibrated(xhistograms, shistograms, path, nthreads=1):
    def helper(quad):
        for ch in range(32):
            fig, ax = _uncalibrated(xhistograms.bins, xhistograms.counts[quad][ch],
                                    shistograms.bins, shistograms.counts[quad][ch],
                                    figsize=(9, 4.5))
            ax.set_title("Uncalibrated plot - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in xhistograms.counts.keys())


def draw_and_save_diagns(histograms, res_fit, path, nthreads=1):
    def helper(quad):
        for ch in res_fit[quad].index:
            fig, ax = _diagnostics(histograms.bins,
                                   histograms.counts[quad][ch],
                                   res_fit[quad].loc[ch].loc[:, 'center'],
                                   res_fit[quad].loc[ch].loc[:, 'amp'],
                                   res_fit[quad].loc[ch].loc[:, 'fwhm'],
                                   res_fit[quad].loc[ch].loc[:, ['lim_low', 'lim_high']].values.reshape(2, -1).T,
                                   figsize=(9, 4.5), dpi=150)
            ax.set_title("Diagnostic plot - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in res_fit.keys())


def draw_and_save_channels_xspectra(histograms, res_cal, radsources:dict, path, nthreads=1):
    def helper(quad):
        for ch in res_cal[quad].index:
            enbins = (histograms.bins - res_cal[quad].loc[ch]['offset']) / res_cal[quad].loc[ch]['gain']
            fig, ax = _spectrum(enbins,
                                histograms.counts[quad][ch],
                                radsources,
                                elims=_compute_lims_for_x(radsources),
                                figsize=(9, 4.5))
            ax.set_title("Spectra plot X - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in res_cal.keys())


def draw_and_save_channels_sspectra(histograms, res_cal, res_slo, radsources: dict, path, nthreads=1):
    def helper(quad):
        for ch in res_slo[quad].index:
            xenbins = (histograms.bins - res_cal[quad].loc[ch]['offset']) / res_cal[quad].loc[ch]['gain']
            enbins = xenbins/res_slo[quad]['light_out'].loc[ch]/PHT_KEV

            fig, ax = _spectrum(enbins,
                                histograms.counts[quad][ch],
                                radsources,
                                elims=_compute_lims_for_s(radsources),
                                figsize=(9, 4.5))
            ax.set_title("Spectra plot S - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in res_slo.keys())


def draw_and_save_xspectrum(calibrated_events, radsources: dict, path):
    if not calibrated_events.empty:
        xevs = calibrated_events[calibrated_events['EVTYPE'] == 'X']
        xcounts, xbins = np.histogram(xevs['ENERGY'], bins=np.arange(2, 40, 0.05))

        fig, ax = _spectrum(xbins,
                            xcounts,
                            radsources,
                            elims=_compute_lims_for_x(radsources),
                            figsize=(9, 4.5))
        ax.set_title("Spectrum X")
        fig.savefig(path)
        plt.close(fig)
        return True


def draw_and_save_sspectrum(calibrated_events, radsources: dict, path):
    if not calibrated_events.empty:
        sevs = calibrated_events[calibrated_events['EVTYPE'] == 'S']
        scounts, sbins = np.histogram(sevs['ENERGY'], bins=np.arange(30, 1000, 2))

        fig, ax = _spectrum(sbins,
                            scounts,
                            radsources,
                            elims=_compute_lims_for_s(radsources),
                            figsize=(9, 4.5))
        ax.set_title("Spectrum S")
        fig.savefig(path)
        plt.close(fig)
        return True


def draw_and_save_lins(res_cal, res_fit, radsources: dict, path, nthreads=1):
    def helper(quad):
        for ch in res_cal[quad].index:
            fig, ax = _linearity(*res_cal[quad].loc[ch][['gain', 'gain_err', 'offset', 'offset_err']],
                                 res_fit[quad].loc[ch].loc[:, 'center'],
                                 res_fit[quad].loc[ch].loc[:, 'center_err'],
                                 radsources,
                                 figsize=(7,7))
            ax[0].set_title("Linearity plot - CH{:02d}Q{}".format(ch, quad))
            fig.savefig(path(quad, ch))
            plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in res_cal.keys())


def draw_and_save_qlooks(res_cal, path, nthreads=1):
    def helper(quad):
        fig, axs = _quicklook(res_cal[quad])
        axs[0].set_title("Calibration quicklook - Quadrant {}".format(quad))
        fig.savefig(path(quad))
        plt.close(fig)

    return Parallel(n_jobs=nthreads)(delayed(helper)(quad) for quad in res_cal.keys())


def _uncalibrated(xbins, xcounts, sbins, scounts, **kwargs):
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.step(sbins[:-1], scounts, color='tomato', label='S events', where='post')
    ax.fill_between(sbins[:-1], scounts, step="post", alpha=0.2, color='tomato')
    ax.step(xbins[:-1], xcounts, label='X events', where='post')
    ax.fill_between(xbins[:-1], xcounts, step="post", alpha=0.2)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Counts")
    ax.set_xlabel("ADU")
    ax.legend()
    return fig, ax


normal = (lambda x, amp, sigma, x0: amp*np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*sqrt(2*pi)))


def _diagnostics(bins, counts, centers, amps, fwhms, limits, **kwargs):
    low_lims, high_lims = [*zip(*limits)]
    min_xlim, max_xlim = min(low_lims) - 500, max(high_lims) + 500
    start = np.where(bins >= min_xlim)[0][0]
    stop = np.where(bins < max_xlim)[0][-1]
    min_ylim, max_ylim = 0, max(counts[start:stop])*1.1

    colors = [plt.cm.tab10(i) for i in range(1, len(limits) + 1)]
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.step(bins[:-1], counts, where='post')
    ax.fill_between(bins[:-1], counts, step="post", alpha=0.4)
    for ctr, amp, fwhm, lims, col in zip(centers, amps, fwhms, limits, colors):
        ax.axvline(ctr, linestyle='dotted')
        ax.axvspan(*lims, color=col, alpha=0.1)
        xs = np.linspace(*lims, 200)
        ax.plot(xs, normal(xs, amp, fwhm/2.355, ctr), color=col)
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
    ax.step(xs, ys, where='post')
    ax.fill_between(xs, ys, step="post", alpha=0.4)
    for key, value, col in zip(radsources_keys, radsources_energies, colors):
        ax.axvline(value, linestyle="dashed", color=col, label=key)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Counts')
    ax.legend(loc="upper right")
    return fig, ax


def _linearity(gain, gain_err, offset, offset_err, adcs, adcs_err, radsources: dict, **kwargs):
    radsources_energies = np.array([l.energy for l in radsources.values()])
    measured_energies_err =  np.sqrt((adcs_err**2)*(1/gain)**2 +
                                     (gain_err**2)*((adcs - offset)/gain**2)**2 +
                                     (offset_err**2)*(1/gain)**2)
    residual = gain * radsources_energies + offset - adcs
    res_err = np.sqrt((gain_err ** 2) * (radsources_energies ** 2) +
                      offset_err ** 2 +
                      adcs_err ** 2)
    perc_residual = 100 * residual / adcs
    perc_residual_err = 100 * res_err / adcs

    prediction_discrepancy = (adcs-offset)/gain - radsources_energies
    perc_prediction_discrepancy = 100 * prediction_discrepancy / radsources_energies
    perc_measured_energies_err = 100 * measured_energies_err / radsources_energies

    margin = (radsources_energies[-1] - radsources_energies[0]) / 10
    xs = np.linspace(radsources_energies[0] - margin, radsources_energies[-1] + margin, 10)

    fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [6, 3, 3]}, sharex=True, **kwargs)
    axs[0].errorbar(radsources_energies, adcs, yerr=adcs_err, fmt='o')
    axs[0].plot(xs, gain * xs + offset)
    axs[1].errorbar(radsources_energies, perc_residual, yerr=perc_residual_err, fmt='o', capsize=5)
    axs[2].errorbar(radsources_energies, perc_prediction_discrepancy, yerr=perc_measured_energies_err, fmt='o', capsize=5)
    axs[0].set_ylabel("Measured Energy [keV]")
    axs[1].set_ylabel("Residual [%]")
    axs[2].set_ylabel("Prediction error [%]")
    axs[2].set_xlabel("Energy [keV]")
    return fig, axs


def _quicklook(calres, **kwargs):
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


def _sloplot(res_slo, **kwargs):
    x = res_slo.index
    y = res_slo['light_out']
    yerr = res_slo['light_out_err']
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
