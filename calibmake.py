import argparse
from os import cpu_count
from collections import namedtuple

import pandas as pd
from pathlib import Path
import logging

from source import upaths
from source import interface
from source.io import pandas_from
from source.wrangle import get_couples
from source.wrangle import add_evtype_tag
from source.wrangle import infer_onchannels
from source.wrangle import filter_delay
from source.wrangle import filter_spurious
from source.specutils import compute_histogram
from source.xpeaks import fit_xradsources
from source.sdds import calibrate_sdds
from source.inventory import fetch_default_sdd_calibration
from source.inventory import radsources_dicts
from source.errors import ModelNotFoundError


START, STOP, STEP = 15000, 28000, 10
NBINS = int((STOP - START) / STEP)
END = START + NBINS * STEP
BINNING = (START, END, NBINS)
RETRIGGER_TIME_IN_S = 20 * (10**-6)

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
        radsources = radsources_dicts(to_list(args.radsources))
        scintillator_couples = get_couples()

    with console.status("Preprocessing.."):
        data, channels = preprocess(data, scintillator_couples, console)
        xhistograms, shistograms = make_histograms(data, BINNING, console)

    with console.status("Calibrating.."):
        xradsources, sradsources = radsources
        if xradsources:
            xfit_results, _ = fit_xradsources(
                xhistograms,
                xradsources,
                channels,
                hint,
            )

            sdd_calibrations, _ = calibrate_sdds(
                xradsources,
                xfit_results,
            )


            pass
    return True


def log_to(filepath):
    logging.basicConfig(
        filename=filepath,
        filemode='w',
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


def fetch_hint(model, temperature, console):
    hint, key = fetch_default_sdd_calibration(model, temperature)
    console.log(":open_book: Loaded detection hints for {}@{}°C.".format(*key))
    return hint


def to_list(radsources_string):
    return radsources_string.upper().split(",")


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


def _to_dfdict(x, idx):
    return {q: pd.DataFrame(x[q], index=idx).T.rename_axis("channel") for q in x.keys()}


if __name__ == "__main__":
    systhreads = min(4, cpu_count())
    args = parse_args()
    log_to(upaths.LOGFILE(args.filepath))

    run(args)
