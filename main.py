import numpy as np
import pandas as pd
from pathlib import Path
from os import cpu_count

from configs import fm1
from source.structs import add_evtype_flag_to, pandas_from
from source.specutilities import detect_peaks, fit_peaks, calibrate_chn
from source.plot import draw_and_save_diagns, draw_and_save_xspectra, draw_and_save_lins, draw_and_save_qlooks
from source.parser import parser
from source import upaths
from source import interface


lines = {'Fe 5.9 keV': 5.9, 'Cd 22.1 keV': 22.1, 'Cd 24.9 keV': 24.9}


asics = 'ABCD'
fit_params = ["center", "center_err", "fwhm", "fwhm_err", "amp", "amp_err", "lim_low", "lim_high"]
cal_params = ["gain", "gain_err", "offset", "offset_err", "chi2"]
start, stop, step = 15000, 24000, 10
nbins = int((stop - start) / step)


def xcalibrate(asics, onchannels, data, lines, start, nbins, step):
    results_fit, results_cal, hists = {}, {}, {}
    lines_keys, lines_values = zip(*lines.items())
    for asic in asics:
        results_fit_asic, results_cal_asic, hist_asic = {}, {}, {}

        couples = fm1.get_couples(asic)
        quad_df = add_evtype_flag_to(data[data['QUADID'] == asic], couples)
        for ch in onchannels[asic]:
            ch_data = quad_df[(quad_df['CHN'] == ch) & (quad_df['EVTYPE'] == 'X')]
            counts, bins = np.histogram(ch_data['ADC'], range=(start, start + nbins * step), bins=nbins)
            limits = detect_peaks(bins, counts, lines_values)
            centers, center_errs, *etc = fit_peaks(bins, counts, limits)
            gain, gain_err, offset, offset_err, chi2 = calibrate_chn(centers, center_errs, lines_values)

            results_fit_asic[ch] = np.concatenate((centers, center_errs, *etc, *limits.T))
            results_cal_asic[ch] = np.array((gain, gain_err, offset, offset_err, chi2))
            hist_asic[ch] = counts
        results_fit[asic] = pd.DataFrame(results_fit_asic, index=pd.MultiIndex.from_product((fit_params, lines_keys))).T
        results_cal[asic] = pd.DataFrame(results_cal_asic, index=cal_params).T
        hists[asic] = hist_asic
    return results_fit, results_cal, (bins, hists)


def write_reports(res_fit, res_cal, path_fit, path_cal):
    for quad, df in res_fit.items():
        df.to_csv(path_fit(quad), sep=';')
    for quad, df in res_cal.items():
        df.to_csv(path_cal(quad), sep=';')
    return True


def get_from(fitspath, console, cache=True):
    cached = upaths.CACHEDIR().joinpath(fitspath.name).with_suffix('.pkl.gz')
    if cached.is_file() and cache:
        out = pd.read_pickle(cached)
        console.log(":white_check_mark: Data were loaded from cache.")
        return out
    elif fitspath.is_file():
        out = pandas_from(fitspath)
        console.log(":white_check_mark: Data loaded.")
        if cache:
            out.to_pickle(cached)
            console.log(":blue_book: Data saved to cache.")
            return out
        return out
    else:
        raise FileNotFoundError('Could not locate input datafile.')


def infer_onchannels(data: pd.DataFrame, asics):
    return {asic: np.unique(data[data['QUADID'] == asic]['CHN']) for asic in asics}


if __name__ == '__main__':
    args = parser.parse_args()
    console = interface.boot()

    with console.status("Building dataset.."):
        console.log(":question_mark: Looking for data..")
        filepath = Path(args.filepath_in)
        cache = not args.nocache
        data = get_from(filepath, console, cache)
        onchannels = infer_onchannels(data, asics)

    tracked = interface.tracked_onchannels(onchannels, asics, console)
    res_fit, res_cal, (bins, histograms) = xcalibrate(asics, tracked, data, lines, start, nbins, step)
    console.log(":white_check_mark: Calibration done!")

    with console.status("Writing and drawing.."):
        write_reports(res_fit, res_cal, upaths.FITREPORT(filepath), upaths.CALREPORT(filepath))
        console.log(":blue_book: Wrote fit and calibration results.")

        nthread = min(4, cpu_count())
        draw_and_save_qlooks(asics, res_cal, upaths.QLKPLOT(filepath))
        console.log(":chart_increasing: Saved quicklook plots.")
        draw_and_save_diagns(asics, onchannels, bins, histograms, res_fit, upaths.DNGPLOT(filepath), nthread)
        console.log(":chart_increasing: Saved fit diagnostics plots.")
        draw_and_save_xspectra(asics, onchannels, bins, histograms, res_cal, lines, upaths.SPEPLOT(filepath), nthread)
        console.log(":chart_increasing: Saved spectra plots.")
        draw_and_save_lins(asics, onchannels, res_cal, res_fit, lines, upaths.LINPLOT(filepath), nthread)
        console.log(":chart_increasing: Saved linearity plots.")

    goodbye = interface.shutdown(console)
