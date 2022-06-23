from os import cpu_count

import pandas as pd
from pathlib import Path

from source.structs import get_couples
from source.structs import pandas_from
from source.structs import infer_onchannels
from source.structs import write_report_to_excel
from source.structs import write_eventlist_to_fits
from source.specutilities import xcalibrate
from source.specutilities import scalibrate
from source.specutilities import histogram
from source.specutilities import add_evtype_tag
from source.plot import draw_and_save_diagns
from source.plot import draw_and_save_channels_xspectra
from source.plot import draw_and_save_channels_sspectra
from source.plot import draw_and_save_sspectrum
from source.plot import draw_and_save_xspectrum
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

########################
import numpy as np

from source.specutilities import PHT_KEV


def make_calibrated_evlist(data, res_cal, res_slo, couples):
    times = np.array([])
    ens = np.array([])
    evtypes = np.array([])
    chns = np.array([])
    quads = np.array([])

    for quad in res_cal.keys():
        quad_data = data[(data['QUADID'] == quad) & (data['CHN'].isin(res_slo[quad].index))]

        xens = (quad_data['ADC'] - res_cal[quad].loc[quad_data['CHN']]['offset'].values) \
                     / res_cal[quad].loc[quad_data['CHN']]['gain'].values
        quad_data.insert(0,'XENS',xens)

        xevents = quad_data[quad_data['EVTYPE'] == 'X']
        xens, xtimes, xchns, xquad, xtype = xevents[['XENS', 'TIME', 'CHN', 'QUADID', 'EVTYPE']].values.T

        sevents = quad_data[quad_data['EVTYPE'] == 'S']
        sevents = sevents.assign(CHN=sevents['CHN'].map(dict(couples[quad])).fillna(sevents['CHN']))
        sevents_grouped = sevents.groupby(['TIME', 'CHN'])
        stimes, schns = np.array([*sevents_grouped.groups.keys()]).T
        schns_comp = pd.Series(schns).map({v: k for k,v in dict(couples[quad]).items()})
        slos = res_slo[quad]['light_out']
        sens = sevents_grouped.sum()['XENS'].values/(slos.loc[schns].values + slos.loc[schns_comp].values)/PHT_KEV

        times = np.concatenate((times, xtimes, stimes))
        ens = np.concatenate((ens, xens, sens))
        chns = np.concatenate((chns, xchns, schns))
        evtypes = np.concatenate((evtypes, xtype, np.array(['S']*len(sens))))
        quads = np.concatenate((quads, xquad, np.array([quad]*len(sens))))

    columns = ['TIME', 'ENERGY', 'EVTYPE', 'CHN', 'QUADID']
    types = ['float64', 'float32', 'string', 'int8', 'string']
    dtypes = {col: tp for col, tp in zip(columns, types)}
    out = pd.DataFrame(np.column_stack((times, ens, evtypes, chns, quads)), columns=columns)
    return out.astype(dtypes).sort_values('TIME').reset_index(drop=True)


########

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

        fm1couples = {q: get_couples('fm1', q) for q in 'ABCD'}
        data = add_evtype_tag(data, couples=fm1couples)
        data = filter_events(data)
        xbins, xhists = histogram(data[data['EVTYPE'] == 'X'], start, nbins, step, nthreads=systhreads)
        sbins, shists = histogram(data[data['EVTYPE'] == 'S'], start, nbins, step, nthreads=systhreads)
        onchannels = infer_onchannels(data)
        console.log(":white_check_mark: Preprocessing done.")

    with console.status("Calibrating.."):
        reshape = (lambda dic, lines: x.reshape(len(fit_params),len(lines)).T.flatten())
        to_dfdict = (lambda x, idx: {q: pd.DataFrame(x[q], index=idx).T for q in x.keys()})

        if xlines:
            _xfitdict, _caldict, xflagged = xcalibrate(xbins, xhists, xlines, onchannels)
            res_xfit = to_dfdict(_xfitdict, pd.MultiIndex.from_product((xlines.keys(), fit_params,)))
            res_cal = to_dfdict(_caldict, cal_params)
            if slines:
                _sfitdict, _slodict, sflagged = scalibrate(sbins, shists, res_cal, slines, lout_guess = (10.,15.))
                res_sfit = to_dfdict(_sfitdict, pd.MultiIndex.from_product((slines.keys(), fit_params,)))
                res_slo = to_dfdict(_slodict, lo_params)
            else:
                res_slo, sflagged = {}, {}
        else:
            res_xfit, res_cal, xflagged = {}, {}, {}
            res_slo, sflagged = {}, {}

        if not res_cal and not res_slo:
            calibrated_events = pd.DataFrame({})
            console.log(":cross_mark: Calibration failed.")
        elif not res_slo or not res_slo:
            calibrated_events = pd.DataFrame({})
            console.log(":yellow_circle: Calibration partially completed. ")
        else:
            calibrated_events = make_calibrated_evlist(data, res_cal, res_slo, fm1couples)
            console.log(":white_check_mark: Calibration complete.")

    with console.status("Writing and drawing.."):
        if res_xfit:
            write_report_to_excel(res_xfit, path=upaths.XFTREPORT(filepath))
        if res_sfit:
            write_report_to_excel(res_sfit, path=upaths.SFTREPORT(filepath))
        if res_cal:
            write_report_to_excel(res_cal, path=upaths.CALREPORT(filepath))
        if res_slo:
            write_report_to_excel(res_slo, path=upaths.SLOREPORT(filepath))
        if not calibrated_events.empty:
            write_eventlist_to_fits(calibrated_events, path=upaths.EVLFITS(filepath))
        if res_cal or res_xfit or res_slo or not calibrated_events.empty:
            console.log(":blue_book: Wrote fit and calibration results.")

        if not args.noplot:
            if draw_and_save_uncalibrated(xbins, xhists, sbins, shists,
                                        path=upaths.UNCPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved uncalibrated plots.")
            if draw_and_save_sspectrum(calibrated_events, slines,
                                        path=upaths.SSPPLOT(filepath)):
                console.log(":chart_increasing: Saved S spectrum.")
            if draw_and_save_xspectrum(calibrated_events, xlines,
                                        path=upaths.XSPPLOT(filepath)):
                console.log(":chart_increasing: Saved X spectrum.")
            if draw_and_save_qlooks(res_cal,
                                        path=upaths.QLKPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved fit quicklooks.")
            if draw_and_save_slo(res_slo,
                                        path=upaths.SLOPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved light-output plots.")
            if draw_and_save_diagns(xbins, xhists, res_xfit,
                                        path=upaths.XDNPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved X fit diagnostics plots.")
            if draw_and_save_diagns(sbins, shists, res_sfit,
                                        path=upaths.SDNPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved S fit diagnostics plots.")
            if draw_and_save_channels_xspectra(xbins, xhists, res_cal, xlines,
                                        path=upaths.XCSPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved X channel spectra plots.")
            if draw_and_save_channels_sspectra(sbins, shists, res_cal, res_slo, slines,
                                        path=upaths.SCSPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved S channel spectra plots.")
            if draw_and_save_lins(res_cal, res_xfit, xlines,
                                        path=upaths.LINPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved linearity plots.")
    console.rule()

    if (xflagged or sflagged) and (res_cal or res_xfit or res_slo):
        flagged = merge_flagged_dicts(xflagged, sflagged)
        console.print("\nWhile processing data I've found {} channels out of {} "
                      "for which calibration could not be completed."
                      .format(sum(len(v) for v in flagged.values()), sum(len(v) for v in onchannels.values())))
        if interface.confirm_prompt("Display flagged channels?"):
            interface.prettyprint(flagged, console=console)

    goodbye = interface.shutdown(console)




