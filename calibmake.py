import argparse
from os import cpu_count
from collections import namedtuple

import pandas as pd
from pathlib import Path
import logging

from source import upaths
from source import interface
from source.io import pandas_from
from source.io import get_writer
from source.io import write_eventlist_to_fits
from source.io import write_report_to_excel
from source.wrangle import get_couples
from source.wrangle import add_evtype_tag
from source.wrangle import infer_onchannels
from source.wrangle import filter_delay
from source.wrangle import filter_spurious
from source.spectra import xcalibrate
from source.spectra import scalibrate
from source.spectra import compute_histogram
from source.spectra import make_events_list
from source.inventory import fetch_default_sdd_calibration
from source.inventory import radsources_dicts
from source.plot import draw_and_save_diagns
from source.plot import draw_and_save_channels_xspectra
from source.plot import draw_and_save_channels_sspectra
from source.plot import draw_and_save_qlooks
from source.plot import draw_and_save_uncalibrated
from source.plot import draw_and_save_slo
from source.plot import draw_and_save_lins
from source.errors import ModelNotFoundError

START, STOP, STEP = 15000, 28000, 10
NBINS = int((STOP - START) / STEP)
END = START + NBINS * STEP
BINNING = (START, END, NBINS)
RETRIGGER_TIME_IN_S = 20 * (10**-6)


FIT_PARAMS = [
    "center",
    "center_err",
    "fwhm",
    "fwhm_err",
    "amp",
    "amp_err",
    "lim_low",
    "lim_high",
]
CAL_PARAMS = [
    "gain",
    "gain_err",
    "offset",
    "offset_err",
    "chi2",
]
LO_PARAMS = [
    "light_out",
    "light_out_err",
]


option = namedtuple("option", ["display", "reply", "promise"])
terminate_mescal = option("Exit mescal.", "So soon?", lambda _: None)
options = [terminate_mescal]


description = (
    "A script to automatically calibrate HERMES-TP/SP "
    "acquisitions of known radioactive sources."
)
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "radsources",
    help="radioactive sources used for calibration. "
    "separated by comma, e.g.:  `-l=Fe,Cd,Cs`. "
    "currently supported sources: Fe, Cd, Cs.",
)
parser.add_argument(
    "filepath",
    help="input acquisition file in standard 0.5 fits format.",
)
parser.add_argument(
    "--cache",
    default=False,
    action="store_true",
    help="enables loading and saving from cache.",
)
parser.add_argument(
    "--fmt",
    default="xslx",
    help="set output format for calibration tables. "
    "supported formats: xslx, csv, fits. "
    "defaults to xslx.",
)
peak_hints_args = parser.add_argument_group(
    "detection hints",
    "to obtain more accurate peak detections a temperature "
    "and a detector model arguments can be specified. "
    "calibrate.py will search through a collection "
    "of established results to get hints "
    "on where to look for a peak. ",
)
peak_hints_args.add_argument(
    "--model",
    "--m",
    help="hermes flight model to calibrate. "
    "supported models: fm1. "
)
peak_hints_args.add_argument(
    "--temperature",
    "--temp",
    "--t",
    type=float,
    help="acquisition temperature in celsius degree. "
    "requires the use of the --model argument",
)


def run(args):

    console = interface.hello()

    with console.status("Initializing.."):
        data = get_from(args.filepath, console, use_cache=args.cache)
        try:
            hint = fetch_hint(args.model, args.temperature, console)
        except ModelNotFoundError:
            hint = None
        radsources = radsources_dicts(radsources_list(args.radsources))
        couples = get_couples()

    with console.status("Preprocessing.."):
        data, channels = preprocess(data, couples, console)
        histograms = make_histograms(data, BINNING, console)

    with console.status("Working on it.."):
        results = calibrate(*histograms, *radsources, channels, hint)
        fits, calibrations, flagged = inspect(*results, console)
        maybe_eventlist = promise(lambda: make_events_list(
            data,
            *calibrations,
            couples,
            systhreads,
        ))

    with console.status("Writing and drawing.."):
        process_results(
            args.filepath,
            histograms,
            radsources,
            fits,
            calibrations,
            maybe_eventlist,
            options,
            args.fmt,
            console,
        )

    if any(flagged):
        warn_about_flagged(flagged, channels, console)

    anything_else(options, console)

    goodbye = interface.shutdown(console)

    return True


def log_to(filepath):
    logging.basicConfig(
        filename=filepath,
        level=logging.INFO,
        format = "[%(funcName)s() @ %(filename)s (L%(lineno)s)] "
        "%(levelname)s: %(message)s"
    )
    return True


def parse_args():
    args = parser.parse_args()
    args.filepath = Path(args.filepath)
    if (args.model is None) and args.temperature:
        parser.error("if a temperature arguments is specified "
                     "a model argument must be specified too ")
    return args


def radsources_list(radsources_string):
    return radsources_string.upper().split(",")


def fetch_hint(model, temperature, console):
    hint, key = fetch_default_sdd_calibration(model, temperature)
    console.log(":open_book: Loaded detection hints for {}@{}°C.".format(*key))
    return hint


def get_from(fitspath: Path, console, use_cache=True):
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


def filter_retrigger(df):
    return filter_delay(df, hold_time=RETRIGGER_TIME_IN_S)


def preprocess(data, detector_couples, console):
    channels = infer_onchannels(data)
    console.log(":white_check_mark: Found active channels.")
    data = add_evtype_tag(data, detector_couples)
    console.log(":white_check_mark: Tagged X and S events.")
    events_pre_filter = len(data)
    data = filter_retrigger(filter_spurious(data))
    filtered_percentual = 100 * (events_pre_filter - len(data)) / events_pre_filter
    console.log(
        ":white_check_mark: Filtered out {:.1f}% of the events.".format(
            filtered_percentual
        )
    )
    return data, channels


def make_histograms(data, binning, console):
    xhistograms = compute_histogram(
        data[data["EVTYPE"] == "X"], *binning, nthreads=systhreads
    )
    shistograms = compute_histogram(
        data[data["EVTYPE"] == "S"], *binning, nthreads=systhreads
    )
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


def calibrate(xhistograms, shistograms, xradsources, sradsources, channels, hint):
    if xradsources:
        _xfitdict, _caldict, xflagged = xcalibrate(
            xhistograms,
            xradsources,
            channels,
            hint,
        )
        index = pd.MultiIndex.from_product((xradsources.keys(), FIT_PARAMS))
        xfit_results = _to_dfdict(_xfitdict, index)
        sdds_calibration = _to_dfdict(_caldict, CAL_PARAMS)

        if sradsources:
            _sfitdict, _slodict, sflagged = scalibrate(
                shistograms,
                sdds_calibration,
                sradsources,
                lout_guess=(10.0, 15.0),
            )
            index = pd.MultiIndex.from_product((sradsources.keys(), FIT_PARAMS))
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


def get_eventlist(el):
    """
    this is for accessing the eventlist.

    Args:
        el: a list (a promise)

    Returns: a dataframe

    """
    car, cdr = el
    if car:
        return cdr
    else: # mutate
        el[0] = 1
        el[1] = cdr()
        return el[1]


def process_results(
    filepath,
    histograms,
    radsources,
    fits,
    calibrations,
    eventlist_promise,
    options,
    fmt,
    console,
):
    xhistograms, shistograms = histograms
    xradsources, sradsources = radsources
    xfit_results, sfit_results = fits
    sdds_calibration, scintillators_lightout = calibrations
    write_report = get_writer(fmt)

    if True:
        options.append(
            _draw_and_save_uncalibrated(
                xhistograms,
                shistograms,
                upaths.UNCPLOT(filepath),
                systhreads,
            )
        )
    if xfit_results:
        options.append(
            _draw_and_save_xdiagns(
                xhistograms,
                xfit_results,
                upaths.XDNPLOT(filepath),
                systhreads,
            )
        )
        options.append(
            _write_xfit_report(
                xfit_results,
                upaths.XFTREPORT(filepath),
            )
        )

    if sdds_calibration:
        write_report(
            sdds_calibration,
            path=upaths.CALREPORT(filepath),
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
                xradsources,
                upaths.XCSPLOT(filepath),
                systhreads,
            )
        )
        options.append(
            _draw_and_save_lins(
                sdds_calibration,
                xfit_results,
                xradsources,
                upaths.LINPLOT(filepath),
                systhreads,
            )
        )

    if sfit_results:
        options.append(
            _draw_and_save_sdiagns(
                shistograms,
                sfit_results,
                upaths.SDNPLOT(filepath),
                systhreads,
            )
        )
        options.append(
            _write_sfit_report(
                sfit_results,
                upaths.SFTREPORT(filepath),
            )
        )

    if scintillators_lightout:
        write_report(
            scintillators_lightout,
            path=upaths.SLOREPORT(filepath),
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
                sradsources,
                upaths.SCSPLOT(filepath),
                systhreads,
            )
        )

    if sdds_calibration and scintillators_lightout:
        options.append(
            _write_eventlist_to_fits(
                lambda: get_eventlist(eventlist_promise),
                upaths.EVLFITS(filepath),
            )
        )
    return True


def merge_flagged_dicts(dx, ds):
    fs = lambda x, l: (*x, "s") if x[0] in l else x
    fx = lambda x, l: (*x, "x") if x[0] in l else x

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
                xhistograms,
                shistograms,
                path,
                nthreads,
            )
        ),
    )


def _draw_and_save_xdiagns(histograms, fit_results, path, nthreads):
    return option(
        display="Save X fit diagnostic plots.",
        reply=":sparkles: Plots saved. :sparkles:",
        promise=promise(
            lambda: draw_and_save_diagns(
                histograms,
                fit_results,
                path,
                nthreads,
            )
        ),
    )


def _draw_and_save_sdiagns(histograms, fit_results, path, nthreads):
    return option(
        display="Save S fit diagnostic plots.",
        reply=":sparkles: Plots saved. :sparkles:",
        promise=promise(
            lambda: draw_and_save_diagns(
                histograms,
                fit_results,
                path,
                nthreads,
            )
        ),
    )


def _write_xfit_report(fit_results, path):
    return option(
        display="Save X fit results.",
        reply=":sparkles: Fit table saved. :sparkles:",
        promise=promise(
            lambda: write_report_to_excel(
                fit_results,
                path,
            )
        ),
    )


def _write_sfit_report(fit_results, path):
    return option(
        display="Save S fit results.",
        reply=":sparkles: Fit table saved. :sparkles:",
        promise=promise(
            lambda: write_report_to_excel(
                fit_results,
                path,
            )
        ),
    )


def _draw_and_save_channels_xspectra(
    xhistograms, sdds_calibration, xradsources, path, nthreads
):
    return option(
        display="Save X channel spectra plots.",
        reply=":sparkles: Plots saved. :sparkles:",
        promise=promise(
            lambda: draw_and_save_channels_xspectra(
                xhistograms,
                sdds_calibration,
                xradsources,
                path,
                nthreads,
            )
        ),
    )


def _draw_and_save_channels_sspectra(
    shistograms,
    sdds_calibration,
    scintillators_lightout,
    sradsources,
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
                sradsources,
                path,
                nthreads,
            )
        ),
    )


def _draw_and_save_lins(sdds_calibration, xfit_results, xradsources, path, nthreads):
    return option(
        display="Save X linearity plots.",
        reply=":sparkles: Plots saved. :sparkles:",
        promise=promise(
            lambda: draw_and_save_lins(
                sdds_calibration,
                xfit_results,
                xradsources,
                path,
                nthreads,
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
    return option(
        display="Write calibrated events to fits.",
        reply=":sparkles: Event list saved. :sparkles:",
        promise=promise(
            lambda: write_eventlist_to_fits(
                thunk(),
                path,
            )
        ),
    )


def promise(f):
    return [0, f]


def fulfill(opt):
    car, cdr = opt
    if car:
        pass
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
                else:
                    console.print("[red]We already did that..")

    interface.print_rule(console)
    return True



if __name__ == "__main__":
    systhreads = min(4, cpu_count())
    args = parse_args()
    log_to(upaths.LOGFILE(args.filepath))

    run(args)
