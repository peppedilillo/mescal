import argparse
import atexit
import logging
from collections import namedtuple
from os import cpu_count
from pathlib import Path

import pandas as pd

from source import interface, upaths
from source.calibration import Calibration
from source.eventlist import (
    add_evtype_tag,
    filter_delay,
    filter_spurious,
    infer_onchannels
)
from source.inventory import get_couples, radsources_dicts
from source.io import (
    get_writer,
    pandas_from,
    write_eventlist_to_fits,
    write_report_to_excel
)
from source.plot import (
    draw_and_save_channels_sspectra,
    draw_and_save_channels_xspectra,
    draw_and_save_diagns,
    draw_and_save_lins,
    draw_and_save_qlooks,
    draw_and_save_slo,
    draw_and_save_spectrum,
    draw_and_save_uncalibrated
)

RETRIGGER_TIME_IN_S = 20 * (10**-6)


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
    "separated by comma, e.g. `Fe,Cd,Cs`."
    "currently supported sources: Fe, Cd, Am, Cs.",
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
    help="hermes flight model to calibrate. " "supported models: fm1. ",
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
        radsources = radsources_dicts(args.radsources)

    with console.status("Preprocessing.."):
        scintillator_couples = get_couples()
        channels = infer_onchannels(data)
        data = preprocess(data, scintillator_couples, console)

    with console.status("Calibrating.."):
        calibration = Calibration(
            channels,
            scintillator_couples,
            radsources,
            args.model,
            args.temperature,
            console,
            systhreads,
        )
        eventlist = calibration(data)

    with console.status("Processing results.."):
        process_results(calibration, eventlist, args.filepath, args.fmt, console)

    if any(calibration.flagged):
        warn_about_flagged(calibration.flagged, channels, console)

    anything_else(options, console)

    goodbye = interface.shutdown(console)

    return True


def start_log(filepath):
    logging.basicConfig(
        filename=filepath,
        filemode="w",
        level=logging.INFO,
        format="[%(funcName)s() @ %(filename)s (L%(lineno)s)] "
        "%(levelname)s: %(message)s",
    )
    return True


@atexit.register
def end_log():
    logging.shutdown()


def parse_args():
    args = parser.parse_args()
    args.filepath = Path(args.filepath)
    args.radsources = args.radsources.upper().split(",")
    if (args.model is None) and args.temperature:
        parser.error(
            "if a temperature arguments is specified "
            "a model argument must be specified too "
        )
    return args


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


def preprocess(data, detector_couples, console):
    data = add_evtype_tag(data, detector_couples)
    console.log(":white_check_mark: Tagged X and S events.")
    events_pre_filter = len(data)
    data = filter_delay(filter_spurious(data), RETRIGGER_TIME_IN_S)
    filtered_percentual = 100 * (events_pre_filter - len(data)) / events_pre_filter
    console.log(
        ":white_check_mark: Filtered out {:.1f}% of the events.".format(
            filtered_percentual
        )
    )
    return data


def warn_about_flagged(flagged, channels, console):
    interface.print_rule(console, "[bold italic]Warning", style="red", align="center")

    for flag in flagged.keys():
        num_flagged = len(flagged[flag])
        message = "{} channels were flagged with '{}'.".format(num_flagged, flag)
        console.print(message)

    num_flagged = len(set([item for sublist in flagged.values() for item in sublist]))
    num_channels = len([item for sublist in channels.values() for item in sublist])
    message = (
        "In total, {} channels out of {} were flagged.\n"
        "For more details, see the log file.".format(num_flagged, num_channels)
    )

    console.print(message)
    return True


# noinspection PyTypeChecker
def process_results(calibration, eventlist, filepath, output_format, console):
    xhistograms = calibration.xhistograms
    shistograms = calibration.shistograms
    xradsources = calibration.get_x_radsources()
    sradsources = calibration.get_gamma_radsources()
    xfit_results = calibration.xfit
    sfit_results = calibration.sfit
    sdd_calibration = calibration.sdd_cal
    effective_louts = calibration.optical_coupling
    write_report = get_writer(output_format)

    if not sdd_calibration and not effective_louts:
        console.log("[bold red]:red_circle: Calibration failed.")
    elif not sdd_calibration or not effective_louts:
        console.log("[bold yellow]:yellow_circle: Calibration partially completed. ")
    else:
        console.log(":green_circle: Calibration complete.")

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

    if sdd_calibration:
        write_report(
            sdd_calibration,
            path=upaths.CALREPORT(filepath),
        )
        console.log(":blue_book: Wrote SDD calibration results.")

        draw_and_save_qlooks(
            sdd_calibration,
            path=upaths.QLKPLOT(filepath),
            nthreads=systhreads,
        )
        console.log(":chart_increasing: Saved X fit quicklook plots.")

        options.append(
            _draw_and_save_channels_xspectra(
                xhistograms,
                sdd_calibration,
                xradsources,
                upaths.XCSPLOT(filepath),
                systhreads,
            )
        )
        options.append(
            _draw_and_save_lins(
                sdd_calibration,
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

    if effective_louts:
        write_report(
            effective_louts,
            path=upaths.SLOREPORT(filepath),
        )
        console.log(":blue_book: Wrote scintillators calibration results.")
        draw_and_save_slo(
            effective_louts,
            path=upaths.SLOPLOT(filepath),
            nthreads=systhreads,
        )
        console.log(":chart_increasing: Saved light-output plots.")

        options.append(
            _draw_and_save_channels_sspectra(
                shistograms,
                sdd_calibration,
                effective_louts,
                sradsources,
                upaths.SCSPLOT(filepath),
                systhreads,
            )
        )

    if sdd_calibration and effective_louts:
        options.append(
            _write_eventlist_to_fits(
                eventlist,
                upaths.EVLFITS(filepath),
            )
        )
        options.append(
            _draw_and_save_spectra(
                eventlist,
                xradsources,
                sradsources,
                upaths.XSPPLOT(filepath),
                upaths.SSPPLOT(filepath),
            )
        )
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


def _write_eventlist_to_fits(eventlist, path):
    return option(
        display="Write calibrated events to fits.",
        reply=":sparkles: Event list saved. :sparkles:",
        promise=promise(
            lambda: write_eventlist_to_fits(
                eventlist,
                path,
            )
        ),
    )


def _draw_and_save_spectra(eventlist, xradsources, sradsources, xpath, spath):
    return option(
        display="Save calibrated spectra.",
        reply=":sparkles: Spectra plot saved :sparkle:",
        promise=promise(
            lambda: draw_and_save_spectrum(
                eventlist,
                xradsources,
                sradsources,
                xpath,
                spath,
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
    args = parse_args()
    systhreads = min(4, cpu_count())
    start_log(upaths.LOGFILE(args.filepath))
    run(args)
