import argparse
from os import cpu_count
from collections import namedtuple

from pathlib import Path

from source import upaths
from source import interface
from source.io import pandas_from
from source.io import write_eventlist_to_fits
from source.wrangle import get_couples
from source.wrangle import add_evtype_tag
from source.wrangle import infer_onchannels
from source.wrangle import filter_delay
from source.wrangle import filter_spurious
from source.spectra import compute_histogram
from source.spectra import make_events_list
from source.inventory import fetch_default_sdd_calibration
from source.inventory import fetch_default_slo_calibration
from source.plot import draw_and_save_channels_xspectra
from source.plot import draw_and_save_channels_sspectra
from source.plot import draw_and_save_uncalibrated


START, STOP, STEP = 15000, 28000, 10
NBINS = int((STOP - START) / STEP)
END = START + NBINS * STEP
BINNING = (START, END, NBINS)
RETRIGGER_TIME_IN_S = 20 * (10**-6)


option = namedtuple("option", ["display", "reply", "promise"])
terminate_mescal = option("Exit mescal.", "So soon?", lambda _: None)
options = [terminate_mescal]


description = (
    "A script to automatically calibrate HERMES-TP/SP "
    "based on previous calibrations."
)
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "filepath",
    help="input acquisition file in standard 0.5 fits format.",
)
parser.add_argument(
    "--m",
    "--model",
    default="fm1",
    help="hermes flight model to calibrate. "
    "supported models: fm1. "
    "defaults to fm1.",
)
parser.add_argument(
    "--t",
    "--temp",
    "--temperature",
    type=float,
    default=20.0,
    help="acquisition temperature in celsius degree. " "defaults to 20.0C",
)


def run():

    console = interface.hello()

    with console.status("Building dataset.."):
        data = get_from(filepath, console)

    with console.status("Preprocessing.."):
        couples = get_couples()
        data, _ = preprocess(data, couples, console)
        histograms = make_histograms(data, BINNING, console)

    with console.status("Writing calibrated event list.."):
        calibrations = sdds_calibration, scintillators_lightout
        process_results(
            filepath,
            couples,
            data,
            histograms,
            calibrations,
            options,
            console,
        )

    anything_else(options, console)

    goodbye = interface.shutdown(console)

    return True


def get_from(fitspath, console):
    console.log(":question_mark: Looking for data..")
    if fitspath.is_file():
        out = pandas_from(fitspath)
        console.log(":open_book: Data loaded.")
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
        data[data["EVTYPE"] == "X"],
        *binning,
        nthreads=systhreads,
    )
    shistograms = compute_histogram(
        data[data["EVTYPE"] == "S"], *binning, nthreads=systhreads
    )
    console.log(":white_check_mark: Binned data.")
    return xhistograms, shistograms


def process_results(
    filepath,
    detector_couples,
    data,
    histograms,
    calibrations,
    options,
    console,
):
    xhistograms, shistograms = histograms
    sdd_calibration, scintillators_lightout = calibrations

    if True:
        options.append(
            _draw_and_save_uncalibrated(
                xhistograms,
                shistograms,
                upaths.UNCPLOT(filepath),
                systhreads,
            )
        )

    if sdds_calibration:
        options.append(
            _draw_and_save_channels_xspectra(
                xhistograms,
                sdds_calibration,
                {},
                upaths.XCSPLOT(filepath),
                systhreads,
            )
        )

    if scintillators_lightout:
        options.append(
            _draw_and_save_channels_sspectra(
                shistograms,
                sdds_calibration,
                scintillators_lightout,
                {},
                upaths.SCSPLOT(filepath),
                systhreads,
            )
        )

    if sdds_calibration and scintillators_lightout:
        event_list = make_events_list(
            data,
            sdds_calibration,
            scintillators_lightout,
            detector_couples,
            systhreads,
        )
        write_eventlist_to_fits(event_list, upaths.EVLFITS(filepath))
        console.log(":blue_book: Wrote calibrated event list.")
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
    args = parser.parse_args()
    filepath = Path(args.filepath)
    model = args.m
    temperature = args.t
    sdds_calibration, _ = fetch_default_sdd_calibration(model, temperature)
    scintillators_lightout, _ = fetch_default_slo_calibration(model, temperature)
    systhreads = min(4, cpu_count())

    run()
