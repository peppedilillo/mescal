from os import cpu_count

import pandas as pd
from pathlib import Path

from source.dataio import get_couples
from source.dataio import pandas_from
from source.dataio import infer_onchannels
from source.dataio import write_report_to_excel
from source.dataio import write_eventlist_to_fits
from source.spectra import xcalibrate
from source.spectra import scalibrate
from source.spectra import histogram
from source.spectra import add_evtype_tag
from source.spectra import make_calibrated_events_list
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

start, stop, step = 15000, 30000, 10
nbins = int((stop - start) / step)
FIT_PARAMS = ["center", "center_err", "fwhm", "fwhm_err", "amp", "amp_err", "lim_low", "lim_high"]
CAL_PARAMS = ["gain", "gain_err", "offset", "offset_err", "chi2"]
LO_PARAMS = ['light_out', 'light_out_err']


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
        console.log(":white_check_mark: Tagged X and S events.")
        data = filter_events(data)
        console.log(":white_check_mark: Applied filters.")
        xbins, xhistograms = histogram(data[data['EVTYPE'] == 'X'], start, nbins, step, nthreads=systhreads)
        sbins, shistograms = histogram(data[data['EVTYPE'] == 'S'], start, nbins, step, nthreads=systhreads)
        console.log(":white_check_mark: Binned data.")
        onchannels = infer_onchannels(data)

    with console.status("Calibrating.."):
        to_dfdict = (lambda x, idx: {q: pd.DataFrame(x[q], index=idx).T for q in x.keys()})

        if xlines:
            _xfitdict, _caldict, xflagged = xcalibrate(xbins, xhistograms, xlines, onchannels)
            xfit_results = to_dfdict(_xfitdict, pd.MultiIndex.from_product((xlines.keys(), FIT_PARAMS,)))
            sdds_calibration = to_dfdict(_caldict, CAL_PARAMS)
            if slines:
                _sfitdict, _slodict, sflagged = scalibrate(sbins, shistograms, sdds_calibration, slines, lout_guess=(10., 15.))
                sfit_results = to_dfdict(_sfitdict, pd.MultiIndex.from_product((slines.keys(), FIT_PARAMS,)))
                scintillators_lightout = to_dfdict(_slodict, LO_PARAMS)
            else:
                sfit_results, scintillators_lightout, sflagged = {}, {}, {}
        else:
            xfit_results, sdds_calibration, xflagged = {}, {}, {}
            sfit_results, scintillators_lightout, sflagged = {}, {}, {}

        if not sdds_calibration and not scintillators_lightout:
            calibrated_events = pd.DataFrame({})
            console.log(":cross_mark: Calibration failed.")
        elif not scintillators_lightout or not scintillators_lightout:
            calibrated_events = pd.DataFrame({})
            console.log(":yellow_circle: Calibration partially completed. ")
        else:
            console.log(":white_check_mark: Calibration complete.")
            calibrated_events = make_calibrated_events_list(data, sdds_calibration, scintillators_lightout, fm1couples)
            console.log(":white_check_mark: Built calibrated event list.")

    with console.status("Writing and drawing.."):
        if xfit_results:
            write_report_to_excel(xfit_results, path=upaths.XFTREPORT(filepath))
        if sfit_results:
            write_report_to_excel(sfit_results, path=upaths.SFTREPORT(filepath))
        if sdds_calibration:
            write_report_to_excel(sdds_calibration, path=upaths.CALREPORT(filepath))
        if scintillators_lightout:
            write_report_to_excel(scintillators_lightout, path=upaths.SLOREPORT(filepath))
        if not calibrated_events.empty:
            write_eventlist_to_fits(calibrated_events, path=upaths.EVLFITS(filepath))
        if sdds_calibration or xfit_results or scintillators_lightout or not calibrated_events.empty:
            console.log(":blue_book: Wrote fit and calibration results.")

        if not args.noplot:

            if draw_and_save_uncalibrated(xbins, xhistograms, sbins, shistograms,
                                          path=upaths.UNCPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved uncalibrated plots.")
            if draw_and_save_sspectrum(calibrated_events, slines,
                                       path=upaths.SSPPLOT(filepath)):
                console.log(":chart_increasing: Saved S spectrum.")
            if draw_and_save_xspectrum(calibrated_events, xlines,
                                       path=upaths.XSPPLOT(filepath)):
                console.log(":chart_increasing: Saved X spectrum.")
            if draw_and_save_qlooks(sdds_calibration,
                                    path=upaths.QLKPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved fit quicklooks.")
            if draw_and_save_slo(scintillators_lightout,
                                 path=upaths.SLOPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved light-output plots.")
            if draw_and_save_diagns(xbins, xhistograms, xfit_results,
                                    path=upaths.XDNPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved X fit diagnostics plots.")
            if draw_and_save_diagns(sbins, shistograms, sfit_results,
                                    path=upaths.SDNPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved S fit diagnostics plots.")
            if draw_and_save_channels_xspectra(xbins, xhistograms, sdds_calibration, xlines,
                                               path=upaths.XCSPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved X channel spectra plots.")
            if draw_and_save_channels_sspectra(sbins, shistograms, sdds_calibration, scintillators_lightout, slines,
                                               path=upaths.SCSPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved S channel spectra plots.")
            if draw_and_save_lins(sdds_calibration, xfit_results, xlines,
                                  path=upaths.LINPLOT(filepath), nthreads=systhreads):
                console.log(":chart_increasing: Saved linearity plots.")
    console.rule()

    if (xflagged or sflagged) and (sdds_calibration or xfit_results or scintillators_lightout):
        flagged = merge_flagged_dicts(xflagged, sflagged)
        console.print("\nWhile processing data I've found {} channels out of {} "
                      "for which calibration could not be completed."
                      .format(sum(len(v) for v in flagged.values()), sum(len(v) for v in onchannels.values())))
        if interface.confirm_prompt("Display flagged channels?"):
            interface.prettyprint(flagged, console=console)

    goodbye = interface.shutdown(console)
