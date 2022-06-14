import numpy as np
import pandas as pd
from pathlib import Path
from os import cpu_count

from configs import fm1
from source.structs import add_evtype_tag, pandas_from
from source.specutilities import xcalibrate, scalibrate, histogram
from source.plot import draw_and_save_diagns, draw_and_save_xspectra, \
    draw_and_save_lins, draw_and_save_qlooks, draw_and_save_uncalibrated, draw_and_save_slo
from source.parser import parser
from source import upaths
from source import interface


xlines = {'Fe 5.9 keV': 5.90, 'Cd 22.1 keV': 22.16, 'Cd 24.9 keV': 24.94}
slines = {'Cs 662 keV': 661.6}

start, stop, step = 15000, 24000, 10
nbins = int((stop - start) / step)
fit_params = ["center", "center_err", "fwhm", "fwhm_err", "amp", "amp_err", "lim_low", "lim_high"]
cal_params = ["gain", "gain_err", "offset", "offset_err", "chi2"]
lo_params = ['light_out', 'light_out_err']


def infer_asics_and_channels(data: pd.DataFrame):
    asics = np.unique(data['QUADID'])
    onchannels = {asic: np.unique(data[data['QUADID'] == asic]['CHN']) for asic in asics}
    return asics, onchannels


def write_reports(result_df, to_path):
    for quad, df in result_df.items():
        df.to_csv(to_path(quad), sep=';')
    return True


def get_from(fitspath, use_cache=True):
    cached = upaths.CACHEDIR().joinpath(fitspath.name).with_suffix('.pkl.gz')
    if cached.is_file() and use_cache:
        out = pd.read_pickle(cached)
        console.log(":white_check_mark: Data were loaded from cache.")
    elif fitspath.is_file():
        out = pandas_from(fitspath)
        console.log(":white_check_mark: Data loaded.")
        if use_cache:
            out.to_pickle(cached)
            console.log(":blue_book: Data saved to cache.")
    else:
        raise FileNotFoundError('could not find input datafile.')
    return out


if __name__ == '__main__':
    args = parser.parse_args()
    console = interface.boot()

    with console.status("Building dataset.."):
        console.log(":question_mark: Looking for data..")
        filepath = Path(args.filepath_in)
        data = get_from(filepath, use_cache=not args.nocache)

    with console.status("Preprocessing.."):
        data = add_evtype_tag(data, couples={q: fm1.get_couples(q) for q in 'ABCD'})
        xbins, xhists = histogram(data[data['EVTYPE'] == 'X'], start, nbins, step)
        sbins, shists = histogram(data[data['EVTYPE'] == 'S'], start, nbins, step)
        _, onchannels = infer_asics_and_channels(data)
        console.log(":white_check_mark: Preprocessing done.")

    with console.status("Calibrating.."):
        # _onchannels = interface.progress_bar(onchannels, log_to=console)
        _fitdf, _caldf, flagged = xcalibrate(xbins, xhists, xlines, onchannels)
        res_fit = {q: pd.DataFrame(_fitdf[q], index=pd.MultiIndex.from_product((fit_params, xlines.keys()))).T
                   for q in _fitdf.keys()}
        res_cal = {q: pd.DataFrame(_caldf[q], index=cal_params).T
                   for q in _fitdf.keys()}
        _res_slo = scalibrate(sbins, shists, res_cal, slines, lout_guess = (10.,15.))
        res_slo = {q: pd.DataFrame(_res_slo[q], index=pd.MultiIndex.from_product((lo_params, slines.keys()))).T
                   for q in _res_slo.keys()}
        console.log(":white_check_mark: Calibration done.")

    with console.status("Writing and drawing.."):
        write_reports(res_fit, to_path=upaths.FITREPORT(filepath))
        write_reports(res_cal, to_path=upaths.CALREPORT(filepath))
        write_reports(res_slo, to_path=upaths.SLOREPORT(filepath))
        console.log(":blue_book: Wrote fit and calibration results.")

        systhreads = min(4, cpu_count())
        if draw_and_save_uncalibrated(xbins, xhists, sbins, shists, to_path=upaths.UNCPLOT(filepath), nthreads=systhreads):
            console.log(":chart_increasing: Saved uncalibrated plots.")
        if draw_and_save_qlooks(res_cal, upaths.QLKPLOT(filepath)):
            console.log(":chart_increasing: Saved fit quicklooks.")
        if draw_and_save_slo(res_slo, upaths.SLOPLOT(filepath), systhreads):
            console.log(":chart_increasing: Saved light-output plots.")
        if draw_and_save_diagns(xbins, xhists, res_fit, upaths.DNGPLOT(filepath), systhreads):
            console.log(":chart_increasing: Saved fit diagnostics plots.")
        if draw_and_save_xspectra(xbins, xhists, res_cal, xlines, upaths.SPEPLOT(filepath), systhreads):
            console.log(":chart_increasing: Saved spectra plots.")
        if draw_and_save_lins(res_cal, res_fit, xlines, upaths.LINPLOT(filepath), systhreads):
            console.log(":chart_increasing: Saved linearity plots.")

    if flagged:
        console.print("\nWhile processing data I've found {} channels I could not calibrate out of {} active channels."
                      .format(sum(len(v) for v in flagged.values()), sum(len(v) for v in onchannels.values())))
        if interface.confirm_prompt("Display flagged channels?"):
            interface.prettyprint(flagged)
    goodbye = interface.shutdown(console)
