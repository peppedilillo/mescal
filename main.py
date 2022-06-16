from os import cpu_count

import pandas as pd
from pathlib import Path

from source.structs import get_couples
from source.structs import pandas_from
from source.structs import infer_onchannels
from source.structs import write_report_to_excel
from source.specutilities import xcalibrate
from source.specutilities import scalibrate
from source.specutilities import histogram
from source.specutilities import add_evtype_tag
from source.plot import draw_and_save_diagns
from source.plot import draw_and_save_xspectra
from source.plot import draw_and_save_qlooks
from source.plot import draw_and_save_uncalibrated
from source.plot import draw_and_save_slo
from source.plot import draw_and_save_lins
from source.parser import compile_sources_dicts
from source.parser import parser
from source import upaths
from source import interface


start, stop, step = 15000, 24000, 10
nbins = int((stop - start) / step)
fit_params = ["center", "center_err", "fwhm", "fwhm_err", "amp", "amp_err", "lim_low", "lim_high"]
cal_params = ["gain", "gain_err", "offset", "offset_err", "chi2"]
lo_params = ['light_out', 'light_out_err']


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


def merge_flagged_dicts(dx, ds):
    fs = (lambda x, l: (*x, 's') if x[0] in l else x)
    fx = (lambda x, l: (*x, 'x') if x[0] in l else x)

    out = {}
    for key in 'ABCD':
        content = sorted(list(set(ds.setdefault(key, []) + dx.setdefault(key, []))))
        if content:
            out[key] = [fs(fx((x,), dx[key]), ds[key]) for x in content]
    return out


if __name__ == '__main__':
    systhreads = min(4, cpu_count())
    args = parser.parse_args()
    xlines, slines = compile_sources_dicts(args.lines)

    console = interface.boot()

    with console.status("Building dataset.."):
        console.log(":question_mark: Looking for data..")
        filepath = Path(args.filepath_in)
        data = get_from(filepath, use_cache=not args.nocache)

    with console.status("Preprocessing.."):
        filter_events = (lambda df: df[(df['NMULT'] < 2) | ((df['NMULT'] == 2) & (df['EVTYPE'] == 'S'))])

        data = add_evtype_tag(data, couples={q: get_couples('fm1', q) for q in 'ABCD'})
        data = filter_events(data)
        xbins, xhists = histogram(data[data['EVTYPE'] == 'X'], start, nbins, step, nthreads=systhreads)
        sbins, shists = histogram(data[data['EVTYPE'] == 'S'], start, nbins, step, nthreads=systhreads)
        onchannels = infer_onchannels(data)
        console.log(":white_check_mark: Preprocessing done.")

    with console.status("Calibrating.."):
        # _onchannels = interface.progress_bar(onchannels, log_to=console)
        if xlines:
            _fitdf, _caldf, xflagged = xcalibrate(xbins, xhists, xlines, onchannels)
            res_fit = {q: pd.DataFrame(_fitdf[q], index=pd.MultiIndex.from_product((fit_params, xlines.keys()))).T
                       for q in _fitdf.keys()}
            res_cal = {q: pd.DataFrame(_caldf[q], index=cal_params).T
                   for q in _fitdf.keys()}
            if slines:
                _res_slo, sflagged = scalibrate(sbins, shists, res_cal, slines, lout_guess = (10.,15.))
                res_slo = {q: pd.DataFrame(_res_slo[q], index=pd.MultiIndex.from_product((lo_params, slines.keys()))).T
                           for q in _res_slo.keys()}
            else:
                res_slo, sflagged = {}, {}

            if not (res_fit or res_cal or res_slo):
                console.log(":cross_mark: Calibration failed.")
            else:
                console.log(":white_check_mark: Calibration complete.")
        else:
            res_fit, res_cal, xflagged = {}, {}, {}
            res_slo, sflagged = {}, {}

    with console.status("Writing and drawing.."):
        if res_cal:
            write_report_to_excel(res_fit, path=upaths.FITREPORT(filepath))
        if res_cal:
            write_report_to_excel(res_cal, path=upaths.CALREPORT(filepath))
        if res_slo:
            write_report_to_excel(res_slo, path=upaths.SLOREPORT(filepath))
        if res_cal or res_fit or res_slo:
            console.log(":blue_book: Wrote fit and calibration results.")


        if draw_and_save_uncalibrated(xbins, xhists, sbins, shists, path=upaths.UNCPLOT(filepath), nthreads=systhreads):
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
    console.rule()

    if (xflagged or sflagged) and (res_cal or res_fit or res_slo):
        flagged = merge_flagged_dicts(xflagged, sflagged)
        console.print("\nWhile processing data I've found {} channels out of {} "
                      "for which calibration could not be completed."
                      .format(sum(len(v) for v in flagged.values()), sum(len(v) for v in onchannels.values())))
        if interface.confirm_prompt("Display flagged channels?"):
            interface.prettyprint(flagged, console=console)

    goodbye = interface.shutdown(console)
