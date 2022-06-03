import numpy as np
import pandas as pd
from pathlib import Path

from configs import fm1
from source.structs import add_evtype_flag_to, infer_onchannels, pandas_from
from source.specutilities import detect_peaks, fit_peaks, calibrate
from source.parser import parser
from source import plot
from source import upaths
from source import interface

asics = 'D'
fit_params = ["center", "center_err", "fwhm", "fwhm_err", "amp", "amp_err", "lim_low", "lim_up"]
cal_params = ["gain", "gain_err", "offset", "offset_err", "chi2"]
lines = {'Fe 5.9 keV': 5.9, 'Cd 22.1 keV': 22.1, 'Cd 24.9 keV': 24.9}
start, stop, step = 15000, 24000, 10
nbins = int((stop - start) / step)


def get_from(fitspath, log=False):
    (lambda l:l.log(":question_mark: Looking for data.." if log else 0))(log)
    cached = upaths.CACHEDIR.joinpath(fitspath.name).with_suffix('.pkl.gz')
    if cached.is_file():
        df = pd.read_pickle(cached)
        (lambda l: l.log(":white_check_mark: Data loaded from cache." if log else 0))(log)
    elif fitspath.is_file():
        df = pandas_from(fitspath)
        (lambda l: l.log(":white_check_mark: Data loaded." if log else 0))(log)
        df.to_pickle(cached)
        (lambda l: l.log(":writing_hand: Saved data to cache." if log else 0))(log)
    else:
        raise FileNotFoundError('Could not locate input datafile.')
    return df


def calibrate_fits(data, lines, start, nbins, step, log=False):
    results_fit, results_cal = {}, {}
    lines_keys, lines_values = zip(*lines.items())
    for asic in asics:
        results_fit_asic, results_cal_asic = {}, {}

        couples = fm1.get_couples(asic)
        onchannels = infer_onchannels(data, asic)
        quad_df = add_evtype_flag_to(data[data['QUADID'] == asic], couples)
        for ch in onchannels:
            ch_data = quad_df[(quad_df['CHN'] == ch) & (quad_df['EVTYPE'] == 'X')]
            counts, bins = np.histogram(ch_data['ADC'], range=(start, start + nbins * step), bins=nbins)
            limits = detect_peaks(bins, counts, lines_values)
            centers, center_errs, *etc = fit_peaks(bins, counts, limits)
            gain, gain_err, offset, offset_err, chi2 = calibrate(centers, center_errs, lines_values)

            results_fit_asic[ch] = np.concatenate((centers, center_errs, *etc, *limits.T))
            results_cal_asic[ch] = np.array((gain, gain_err, offset, offset_err, chi2))
        results_fit[asic] = pd.DataFrame(results_fit_asic, index=pd.MultiIndex.from_product((fit_params, lines_keys))).T
        results_cal[asic] = pd.DataFrame(results_cal_asic, index=cal_params).T
        (lambda l: l.log(":white_check_mark: Calibration ASIC {} done!".format(asic) if log else 0))(log)
    return results_fit, results_cal


if __name__ == '__main__':
    args = parser.parse_args()
    console = interface.boot()

    with console.status("Building dataset.."):
        filepath = Path(args.filepath_in)
        data = get_from(filepath, log=console)
    with console.status("Running calibration.."):
        res_fit, res_cal = calibrate_fits(data,lines,start, nbins, step, log=console)
    with console.status("Storing results.."):
        for quad, df in res_fit.items():
            df.to_csv(upaths.FITREPORT(filepath, quad), sep =';')
        for quad, df in res_cal.items():
            df.to_csv(upaths.CALREPORT(filepath, quad), sep =';')

    goodbye = interface.shutdown(console)