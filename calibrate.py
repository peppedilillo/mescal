from os import cpu_count
from collections import namedtuple

import pandas as pd
from pathlib import Path

from source.dataio import get_couples
from source.dataio import pandas_from
from source.dataio import infer_onchannels
from source.dataio import write_report_to_excel
from source.dataio import write_eventlist_to_fits
from source.spectra import xcalibrate
from source.spectra import scalibrate
from source.spectra import compute_histogram
from source.spectra import add_evtype_tag
from source.spectra import make_events_list
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


START, STOP, STEP = 15000, 30000, 10
NBINS = int((STOP - START) / STEP)

FIT_PARAMS = ["center", "center_err", "fwhm", "fwhm_err", "amp", "amp_err", "lim_low", "lim_high"]
CAL_PARAMS = ["gain", "gain_err", "offset", "offset_err", "chi2"]
LO_PARAMS = ['light_out', 'light_out_err']

option = namedtuple('option', ['display', 'reply', 'action', 'args', 'kwargs'], defaults=[(), {}])
terminate_mescal = option('Goodbye.', 'So soon?', (lambda _: None))
options = [terminate_mescal]


def get_from(fitspath, use_cache=True):
    cached = upaths.CACHEDIR().joinpath(fitspath.name).with_suffix('.pkl.gz')
    if cached.is_file() and use_cache:
        out = pd.read_pickle(cached)
        console.log(":exclamation_mark: Data were loaded from cache.")
    elif fitspath.is_file():
        out = pandas_from(fitspath)
        console.log(":open_book: Data loaded.")
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

    ####################################################################################################################
    with console.status("Building dataset.."):
        console.log(":question_mark: Looking for data..")
        filepath = Path(args.filepath_in)
        data = get_from(filepath, use_cache=not args.nocache)

    with console.status("Preprocessing.."):
        fm1couples = {q: get_couples('fm1', q) for q in 'ABCD'}
        data = add_evtype_tag(data, couples=fm1couples)
        console.log(":white_check_mark: Tagged X and S events.")
        filter_events = (lambda df: df[(df['NMULT'] < 2) | ((df['NMULT'] == 2) & (df['EVTYPE'] == 'S'))])
        data = filter_events(data)
        console.log(":white_check_mark: Applied filters.")
        binning = START, NBINS, STEP
        xhistograms = compute_histogram(data[data['EVTYPE'] == 'X'], *binning, nthreads=systhreads)
        shistograms = compute_histogram(data[data['EVTYPE'] == 'S'], *binning, nthreads=systhreads)
        console.log(":white_check_mark: Binned data.")
        onchannels = infer_onchannels(data)
        console.log(":white_check_mark: Found active channels.")

    with console.status("Calibrating.."):
        to_dfdict = (lambda x, idx: {q: pd.DataFrame(x[q], index=idx).T for q in x.keys()})

        if xlines:
            _xfitdict, _caldict, xflagged = xcalibrate(xhistograms,
                                                       xlines,
                                                       onchannels)
            xfit_results = to_dfdict(_xfitdict, pd.MultiIndex.from_product((xlines.keys(), FIT_PARAMS,)))
            sdds_calibration = to_dfdict(_caldict, CAL_PARAMS)
            if slines:
                _sfitdict, _slodict, sflagged = scalibrate(shistograms,
                                                           sdds_calibration,
                                                           slines,
                                                           lout_guess=(10., 15.))
                sfit_results = to_dfdict(_sfitdict, pd.MultiIndex.from_product((slines.keys(), FIT_PARAMS,)))
                scintillators_lightout = to_dfdict(_slodict, LO_PARAMS)
            else:
                sfit_results, scintillators_lightout, sflagged = {}, {}, {}
        else:
            xfit_results, sdds_calibration, xflagged = {}, {}, {}
            sfit_results, scintillators_lightout, sflagged = {}, {}, {}

        if not sdds_calibration and not scintillators_lightout:
            console.log(":cross_mark: Calibration failed.")
        elif not scintillators_lightout or not scintillators_lightout:
            console.log(":yellow_circle: Calibration partially completed. ")
        else:
            console.log(":white_check_mark: Calibration complete.")

    with console.status("Writing and drawing.."):
        if True:
            options.append(option(display="Save uncalibrated plots.",
                                  reply=":sparkles: Saved uncalibrated plots. :sparkles:",
                                  action=draw_and_save_uncalibrated,
                                  args=(xhistograms, shistograms),
                                  kwargs={'path': upaths.UNCPLOT(filepath),
                                          'nthreads': systhreads}))
        if xfit_results:
            options.append(option(display="Save X fit diagnostic plots.",
                                  reply=":sparkles: Plots saved. :sparkles:",
                                  action=draw_and_save_diagns,
                                  args=(xhistograms, xfit_results),
                                  kwargs={'path': upaths.XDNPLOT(filepath),
                                          'nthreads': systhreads}))
            options.append(option(display="Save X fit results.",
                                  reply=":sparkles: Fit table saved. :sparkles:",
                                  action=write_report_to_excel,
                                  args=(xfit_results,),
                                  kwargs={'path': upaths.XFTREPORT(filepath)}))

        if sdds_calibration:
            write_report_to_excel(sdds_calibration, path=upaths.CALREPORT(filepath))
            console.log(":blue_book: Wrote SDD calibration results.")
            draw_and_save_qlooks(sdds_calibration, path=upaths.QLKPLOT(filepath), nthreads=systhreads)
            console.log(":chart_increasing: Saved X fit quicklook plots.")

            options.append(option(display="Save X channel spectra plots.",
                                  reply=":sparkles: Plots saved. :sparkles:",
                                  action=draw_and_save_channels_xspectra,
                                  args=(xhistograms, sdds_calibration, xlines),
                                  kwargs={'path': upaths.XCSPLOT(filepath),
                                          'nthreads': systhreads}))
            options.append(option(display="Save X linearity plots.",
                                  reply=":sparkles: Plots saved. :sparkles:",
                                  action=draw_and_save_lins,
                                  args=(sdds_calibration, xfit_results, xlines),
                                  kwargs={'path': upaths.LINPLOT(filepath),
                                          'nthreads': systhreads}))

        if sfit_results:
            options.append(option(display="Save S fit diagnostic plots.",
                                  reply=":sparkles: Plots saved. :sparkles:",
                                  action=draw_and_save_diagns,
                                  args=(shistograms, sfit_results),
                                  kwargs={'path': upaths.SDNPLOT(filepath),
                                          'nthreads': systhreads}))
            options.append(option(display="Save S fit results.",
                                  reply=":sparkles: Fit table saved. :sparkles:",
                                  action=write_report_to_excel,
                                  args=(sfit_results,),
                                  kwargs={'path': upaths.SFTREPORT(filepath)}))

        if scintillators_lightout:
            write_report_to_excel(scintillators_lightout, path=upaths.SLOREPORT(filepath))
            console.log(":blue_book: Wrote scintillators calibration results.")
            draw_and_save_slo(scintillators_lightout, path=upaths.SLOPLOT(filepath), nthreads=systhreads)
            console.log(":chart_increasing: Saved light-output plots.")

            options.append(option(display="Save S channel spectra plots.",
                                  reply=":sparkles: Plots saved. :sparkles:",
                                  action=draw_and_save_channels_sspectra,
                                  args=(shistograms, sdds_calibration, scintillators_lightout, slines),
                                  kwargs={'path': upaths.SCSPLOT(filepath),
                                          'nthreads': systhreads}))

        if sdds_calibration and scintillators_lightout:
            write_eventlist_to_fits_ = (lambda *args: write_eventlist_to_fits(
                                                        make_events_list(*args,
                                                                         nthreads=systhreads),
                                                        upaths.EVLFITS(filepath)))
            options.append(option(display="Write calibrated events to fits.",
                                  reply=":sparkles: Event list saved. :sparkles:",
                                  action=write_eventlist_to_fits_,
                                  args=(data, sdds_calibration, scintillators_lightout, fm1couples)))

    ####################################################################################################################

    if xflagged or sflagged:
        interface.print_rule(console, "[bold italic]Warning", style='red', align='center')
        flagged = merge_flagged_dicts(xflagged, sflagged)
        console.print(interface.flagged_message(flagged, onchannels))

    ####################################################################################################################

    interface.print_rule(console, "[italic]Optional Outputs", align='center')
    console.print(interface.options_message(options))
    while True:
        answer = interface.prompt_execute_option(options)
        if answer:
            with console.status("Working.."):
                option = options[answer]
                option.action(*option.args, **option.kwargs)
                console.print(option.reply + '\n')
        else:
            console.print(terminate_mescal.reply)
            break
    interface.print_rule(console)

    goodbye = interface.shutdown(console)
