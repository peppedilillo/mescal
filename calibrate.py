from os import cpu_count
from collections import namedtuple

import pandas as pd
from pathlib import Path

from source.io import pandas_from
from source.io import write_eventlist_to_fits
from source.io import write_report_to_excel
from source.wrangle import get_couples
from source.wrangle import add_evtype_tag
from source.wrangle import infer_onchannels
from source.wrangle import filter_delay
from source.spectra import xcalibrate
from source.spectra import scalibrate
from source.spectra import compute_histogram
from source.spectra import make_events_list
from source.plot import draw_and_save_diagns
from source.plot import draw_and_save_channels_xspectra
from source.plot import draw_and_save_channels_sspectra
from source.plot import draw_and_save_qlooks
from source.plot import draw_and_save_uncalibrated
from source.plot import draw_and_save_slo
from source.plot import draw_and_save_lins
from source.parser import compile_sources_dicts
from source.parser import get_writer
from source.parser import parser
from source import upaths
from source import interface


START, STOP, STEP = 15000, 28000, 10
NBINS = int((STOP - START) / STEP)
BINNING = (START, NBINS, STEP)
RETRIGGER_TIME_IN_S = 50 * (10**-6)

FIT_PARAMS = [
    "center",
    "center_err",
    "fwhm",
    "fwhm_err",
    "amp",
    "amp_err",
    "lim_low",
    "lim_high"
]
CAL_PARAMS = [
    "gain",
    "gain_err",
    "offset",
    "offset_err",
    "chi2"
]
LO_PARAMS = [
    "light_out",
    "light_out_err"
]

option = namedtuple("option", ["display", "reply", "promise"])
terminate_mescal = option("Exit mescal.", "So soon?", lambda _: None)
options = [terminate_mescal]


def run():
    console = interface.hello()

    with console.status("Building dataset.."):
        data = get_from(filepath, console, use_cache=args.cache)

    with console.status("Preprocessing.."):
        couples = get_couples("fm1")
        data, channels = preprocess(data, couples, console)
        histograms = make_histograms(data, BINNING, console)

    with console.status("Calibrating.."):
        calibration = calibrate(*histograms, *lines, channels)
        *results, flagged = inspect(*calibration, console)

    with console.status("Writing and drawing.."):
        process_results(filepath, couples, data, histograms, lines, results, options, console)

    if any(flagged):
        warn_about_flagged(flagged, channels, console)

    anything_else(options, console)

    goodbye = interface.shutdown(console)

    return True


def get_from(fitspath, console, use_cache=True):
    console.log(":question_mark: Looking for data..")
    cached = upaths.CACHEDIR().joinpath(fitspath.name).with_suffix(".pkl.gz")
    if cached.is_file() and use_cache:
        out = pd.read_pickle(cached)
        console.log("[bold red]:exclamation_mark: Data were loaded from cache.")
    elif fitspath.is_file():
        out = pandas_from(fitspath)
        console.log(":open_book: Data loaded.")
        if use_cache:
            out.to_pickle(cached)
            console.log(":blue_book: Data saved to cache.")
    else:
        raise FileNotFoundError("could not find input datafile.")
    return out


def preprocess(data, detector_couples, console):
    channels = infer_onchannels(data)
    console.log(":white_check_mark: Found active channels.")
    data = add_evtype_tag(data, detector_couples)
    console.log(":white_check_mark: Tagged X and S events.")
    events_pre_filter = len(data)
    remove_retrigger_and = (lambda df: filter_delay(df, hold_time=RETRIGGER_TIME_IN_S))
    spurious_from = (lambda df: df[(df["NMULT"] < 2) | ((df["NMULT"] == 2) & (df["EVTYPE"] == "S"))])
    data = remove_retrigger_and(spurious_from(data))
    console.log(":white_check_mark: Filtered out {:.1f}% of the events.".format(100*(events_pre_filter - len(data))/events_pre_filter))
    return data, channels


def make_histograms(data, binning, console):
    xhistograms = compute_histogram(data[data["EVTYPE"] == "X"], *binning, nthreads=systhreads)
    shistograms = compute_histogram(data[data["EVTYPE"] == "S"], *binning, nthreads=systhreads)
    console.log(":white_check_mark: Binned data.")
    return xhistograms, shistograms


def inspect(fits, calibrations, flagged, console):
    sdds_calibration, scintillators_lightout = calibrations
    flagged = merge_flagged_dicts(*flagged)

    if not sdds_calibration and not scintillators_lightout:
        console.log("[bold red] :cross_mark: Calibration failed.")
    elif not scintillators_lightout or not scintillators_lightout:
        console.log("[bold yellow]:yellow_circle: Calibration partially completed. ")
    else:
        console.log(":white_check_mark: Calibration complete.")
    return fits, calibrations, flagged


def _to_dfdict(x, idx):
    return {q: pd.DataFrame(x[q], index=idx).T for q in x.keys()}


def calibrate(xhistograms, shistograms, xlines, slines, channels):

    if xlines:
        _xfitdict, _caldict, xflagged = xcalibrate(xhistograms, xlines, channels, calibration_hint)
        index = pd.MultiIndex.from_product(
            (
                xlines.keys(),
                FIT_PARAMS,
            )
        )
        xfit_results = _to_dfdict(_xfitdict, index)
        sdds_calibration = _to_dfdict(_caldict, CAL_PARAMS)
        if slines:
            _sfitdict, _slodict, sflagged = scalibrate(
                shistograms, sdds_calibration, slines, lout_guess=(10.0, 15.0)
            )
            index = pd.MultiIndex.from_product(
                (
                    slines.keys(),
                    FIT_PARAMS,
                )
            )
            sfit_results = _to_dfdict(_sfitdict, index)
            scintillators_lightout = _to_dfdict(_slodict, LO_PARAMS)
        else:
            sfit_results, scintillators_lightout, sflagged = {}, {}, {}
    else:
        xfit_results, sdds_calibration, xflagged = {}, {}, {}
        sfit_results, scintillators_lightout, sflagged = {}, {}, {}

    return (
        (xfit_results, sfit_results),
        (sdds_calibration, scintillators_lightout),
        (xflagged, sflagged),
    )


def process_results(
    filepath,
    detector_couples,
    data,
    histograms,
    lines,
    results,
    options,
    console,
):
    xhistograms, shistograms = histograms
    xlines, slines = lines
    (xfit_results, sfit_results), (
        sdds_calibration,
        scintillators_lightout,
    ) = results

    if True:
        options.append(
            _draw_and_save_uncalibrated(
                xhistograms, shistograms, upaths.UNCPLOT(filepath), systhreads
            )
        )
    if xfit_results:
        options.append(
            _draw_and_save_xdiagns(
                xhistograms, xfit_results, upaths.XDNPLOT(filepath), systhreads
            )
        )
        options.append(
            _write_xfit_report(
                xfit_results, upaths.XFTREPORT(filepath)
            )
        )

    if sdds_calibration:
        write_report(
            sdds_calibration, path=upaths.CALREPORT(filepath)
        )
        console.log(":blue_book: Wrote SDD calibration results.")
        draw_and_save_qlooks(
            sdds_calibration,
            path=upaths.QLKPLOT(filepath),
            nthreads=systhreads,
        )
        console.log(":chart_increasing: Saved X fit quicklook plots.")

        options.append(
            _draw_and_save_channels_xspectra(
                xhistograms,
                sdds_calibration,
                xlines,
                upaths.XCSPLOT(filepath),
                systhreads,
            )
        )
        options.append(
            _draw_and_save_lins(
                sdds_calibration,
                xfit_results,
                xlines,
                upaths.LINPLOT(filepath),
                systhreads,
            )
        )

    if sfit_results:
        options.append(
            _draw_and_save_sdiagns(
                shistograms, sfit_results, upaths.SDNPLOT(filepath), systhreads
            )
        )
        options.append(
            _write_sfit_report(
                sfit_results, upaths.SFTREPORT(filepath)
            )
        )

    if scintillators_lightout:
        write_report(
            scintillators_lightout,
            path=upaths.SLOREPORT(filepath)
        )
        console.log(":blue_book: Wrote scintillators calibration results.")
        draw_and_save_slo(
            scintillators_lightout,
            path=upaths.SLOPLOT(filepath),
            nthreads=systhreads,
        )
        console.log(":chart_increasing: Saved light-output plots.")

        options.append(
            _draw_and_save_channels_sspectra(
                shistograms,
                sdds_calibration,
                scintillators_lightout,
                slines,
                upaths.SCSPLOT(filepath),
                systhreads,
            )
        )

    if sdds_calibration and scintillators_lightout:
        options.append(
            _write_eventlist_to_fits(
                lambda: make_events_list(
                    data,
                    sdds_calibration,
                    scintillators_lightout,
                    detector_couples,
                ),
                upaths.EVLFITS(filepath),
            )
        )
    return True


def merge_flagged_dicts(dx, ds):
    fs = (lambda x, l: (*x, "s") if x[0] in l else x)
    fx = (lambda x, l: (*x, "x") if x[0] in l else x)

    out = {}
    for key in "ABCD":
        content = sorted(list(set(ds.setdefault(key, []) + dx.setdefault(key, []))))
        if content:
            out[key] = [fs(fx((x,), dx[key]), ds[key]) for x in content]
    return out


def warn_about_flagged(flagged, channels, console):
    interface.print_rule(console, "[bold italic]Warning", style="red", align="center")
    console.print(interface.flagged_message(flagged, channels))
    return True


def _draw_and_save_uncalibrated(xhistograms, shistograms, path, nthreads):
    return option(
        display="Save uncalibrated plots.",
        reply=":sparkles: Saved uncalibrated plots. :sparkles:",
        promise=promise(
            lambda: draw_and_save_uncalibrated(
                xhistograms, shistograms, path, nthreads
            )
        ),
    )


def _draw_and_save_xdiagns(histograms, fit_results, path, nthreads):
    return option(
        display="Save X fit diagnostic plots.",
        reply=":sparkles: Plots saved. :sparkles:",
        promise=promise(
            lambda: draw_and_save_diagns(
                histograms, fit_results, path, nthreads
            )
        ),
    )


def _draw_and_save_sdiagns(histograms, fit_results, path, nthreads):
    return option(
        display="Save S fit diagnostic plots.",
        reply=":sparkles: Plots saved. :sparkles:",
        promise=promise(
            lambda: draw_and_save_diagns(
                histograms, fit_results, path, nthreads
            )
        ),
    )


def _write_xfit_report(fit_results, path):
    return option(
        display="Save X fit results.",
        reply=":sparkles: Fit table saved. :sparkles:",
        promise=promise(
            lambda: write_report_to_excel(fit_results, path)
        ),
    )


def _write_sfit_report(fit_results, path):
    return option(
        display="Save S fit results.",
        reply=":sparkles: Fit table saved. :sparkles:",
        promise=promise(
            lambda: write_report_to_excel(fit_results, path)
        ),
    )


def _draw_and_save_channels_xspectra(
    xhistograms, sdds_calibration, xlines, path, nthreads
):
    return option(
        display="Save X channel spectra plots.",
        reply=":sparkles: Plots saved. :sparkles:",
        promise=promise(
            lambda: draw_and_save_channels_xspectra(
                xhistograms, sdds_calibration, xlines, path, nthreads
            )
        ),
    )


def _draw_and_save_channels_sspectra(
    shistograms,
    sdds_calibration,
    scintillators_lightout,
    slines,
    path,
    nthreads,
):
    return option(
        display="Save S channel spectra plots.",
        reply=":sparkles: Plots saved. :sparkles:",
        promise=promise(
            lambda: draw_and_save_channels_sspectra(
                shistograms,
                sdds_calibration,
                scintillators_lightout,
                slines,
                path,
                nthreads,
            )
        ),
    )


def _draw_and_save_lins(
    sdds_calibration, xfit_results, xlines, path, nthreads
):
    return option(
        display="Save X linearity plots.",
        reply=":sparkles: Plots saved. :sparkles:",
        promise=promise(
            lambda: draw_and_save_lins(
                sdds_calibration, xfit_results, xlines, path, nthreads
            )
        ),
    )


def _write_eventlist_to_fits(thunk, path):
    """
    designed to delay the evaluation of make_events_list. call it like :
    _write_eventlist_to_fits(
            (lambda: make_events_list(data,
                                      sdds_calibration,
                                      scintillators_lightout,
                                      detector_couples)),
            upaths.EVLFITS(filepath))
    """
    return option(display="Write calibrated events to fits.",
                  reply=":sparkles: Event list saved. :sparkles:",
                  promise=promise(lambda: write_eventlist_to_fits(thunk(), path)))


def promise(f):
    return [0, f]


def fulfill(opt):
    car, cdr = opt
    if car:
        print("We already did that..")
    else:
        opt[0] = 1
        return cdr()


def anything_else(options, console):
    interface.print_rule(console, "[italic]Optional Outputs", align="center")
    console.print(interface.options_message(options))

    while True:
        answer = interface.prompt_user_about(options)
        if answer is terminate_mescal:
            console.print(terminate_mescal.reply)
            break
        else:
            with console.status("Working.."):
                if fulfill(answer.promise):
                    console.print(answer.reply)

    interface.print_rule(console)
    return True


if __name__ == "__main__":
    systhreads = min(4, cpu_count())
    args = parser.parse_args()
    filepath = Path(args.filepath_in)
    lines = compile_sources_dicts(args.lines)
    write_report = get_writer(args.fmt)

    from source.io import read_report_from_excel
    cal_hint = Path(r"D:\Dropbox\Progetti\mescal\Hermes-Cal-SW\assets\default_calibrations\fm1\20220616_55Fe109Cd137Cs_m20deg_thr105_LV0d5\report_cal.xlsx")
    calibration_hint = read_report_from_excel(cal_hint)
    run()
