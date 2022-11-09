import argparse
import atexit
import logging
import configparser
import sys
from collections import namedtuple
from os import cpu_count
from pathlib import Path
import cmd
import pandas as pd

from source import cmd
from source import interface, upaths
from source.calibration import Calibration
from source.eventlist import (
    add_evtype_tag,
    filter_delay,
    filter_spurious,
    infer_onchannels,
)
from source.inventory import get_couples, radsources_dicts
from source.io import (
    get_writer,
    pandas_from_LV0d5,
    write_eventlist_to_fits,
    write_report_to_excel,
)
from source.plot import (
    draw_and_save_channels_sspectra,
    draw_and_save_channels_xspectra,
    draw_and_save_diagns,
    draw_and_save_lins,
    draw_and_save_qlooks,
    draw_and_save_slo,
    draw_and_save_calibrated_spectra,
    draw_and_save_uncalibrated,
)

RETRIGGER_TIME_IN_S = 20 * (10**-6)

description = (
    "A script to automatically calibrate HERMES-TP/SP "
    "acquisitions of known radioactive sources."
)
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "model",
    choices=["fm1", "pfm", "fm2"],
    help="hermes flight model to calibrate. "
    "supported models: fm1, pfm, fm2.",
)
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
    "--configuration",
    choices=["default", "CAEN-DT5740"],
    default="default",
    help="select which configuration to use."
    "default is designed for the integrated PL",
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


class MescalShell(cmd.Cmd):
    intro = (
        "This is [bold purple]mescal[/] new shell. "
        "Type help or ? to list commands.\n"
    )
    prompt = "[mescal] "
    failure = "[red]Cannnot execute with present calibration."

    def __init__(self, console, filename, calibration, threads):
        super().__init__(console)
        self.calibration = calibration
        self.filename = filename
        self.threads = threads

    def do_save_rawhist_plots(self, arg):
        """Save raw acquisition histogram plots."""
        with self.console.status("Working.."):
            draw_and_save_uncalibrated(
                self.calibration.xhistograms,
                self.calibration.shistograms,
                upaths.UNCPLOT(self.filename),
                self.threads,
            )

    def do_save_xdiagns_plots(self, arg):
        """Save X peak detection diagnostics plots."""
        if not self.calibration.xfit:
            self.console.print(self.failure)
            return False
        with self.console.status("Working.."):
            draw_and_save_diagns(
                self.calibration.xhistograms,
                self.calibration.xfit,
                upaths.XDNPLOT(self.filename),
                self.threads,
            )

    def do_save_xfit_table(self, arg):
        """Save X fit tables."""
        if not self.calibration.xfit:
            self.console.print(self.failure)
            return False
        with self.console.status("Working.."):
            write_report_to_excel(
                self.calibration.xfit,
                upaths.XFTREPORT(self.filename),
            )

    def do_save_xspectra_plots(self, arg):
        """Save X sources fit tables."""
        if not self.calibration.sdd_cal:
            self.console.print(self.failure)
            return False
        with self.console.status("Working.."):
            draw_and_save_channels_xspectra(
                self.calibration.xhistograms,
                self.calibration.sdd_cal,
                self.calibration.xradsources(),
                upaths.XCSPLOT(self.filename),
                self.threads,
            )

    def do_save_xlin_plots(self, args):
        """Save SDD linearity plots."""
        if not self.calibration.sdd_cal:
            self.console.print(self.failure)
            return False
        with self.console.status("Working.."):
            draw_and_save_lins(
                self.calibration.sdd_cal,
                self.calibration.xfit,
                self.calibration.xradsources(),
                upaths.LINPLOT(self.filename),
                self.threads,
            )

    def do_save_sdiagns_plots(self, args):
        """Save S peak detection diagnostics plots."""
        if not self.calibration.sfit:
            self.console.print(self.failure)
            return False
        with self.console.status("Working.."):
            draw_and_save_diagns(
                self.calibration.shistograms,
                self.calibration.sfit,
                upaths.SDNPLOT(self.filename),
                self.threads,
            )

    def do_save_sfit_table(self, arg):
        """Save gamma sources fit tables."""
        if not self.calibration.sfit:
            self.console.print(self.failure)
            return False
        with self.console.status("Working.."):
            write_report_to_excel(
                self.calibration.sfit,
                upaths.SFTREPORT(self.filename),
            )

    def do_save_sspectra_plots(self, arg):
        """Save gamma sources fit tables."""
        if not self.calibration.optical_coupling:
            self.console.print(self.failure)
            return False
        with self.console.status("Working.."):
            draw_and_save_channels_sspectra(
                self.calibration.shistograms,
                self.calibration.sdd_cal,
                self.calibration.optical_coupling,
                self.calibration.sradsources(),
                upaths.SCSPLOT(self.filename),
                self.threads,
            )

    def do_save_event_fits(self, arg):
        """Save calibrated events to fits file."""
        if not self.calibration.eventlist:
            self.console.print(self.failure)
            return False
        with self.console.status("Working.."):
            write_eventlist_to_fits(
                self.calibration.eventlist,
                upaths.EVLFITS(self.filename),
            )

    def do_quit(self, arg):
        """Quit mescal."""
        return True


def run(args):
    console = interface.hello()
    config_dict = unpack_configuration(args.configuration)

    with console.status("Initializing.."):
        data = get_from(args.filepath, console, use_cache=args.cache)
        radsources = radsources_dicts(args.radsources)

    with console.status("Preprocessing.."):
        scintillator_couples = get_couples(args.model)
        channels = infer_onchannels(data)
        data = preprocess(data, scintillator_couples, console)

    with console.status("Calibrating.."):
        calibrated = Calibration(
            channels,
            scintillator_couples,
            radsources,
            detector_model=args.model,
            configuration=config_dict,
            console=console,
            nthreads=systhreads,
        )
        eventlist = calibrated(data)

    with console.status("Processing results.."):
        process_results(
            calibrated, eventlist, args.filepath, args.fmt, console
        )

    if any(calibrated.flagged):
        warn_about_flagged(calibrated.flagged, channels, console)

    shell = MescalShell(console, args.filepath, calibrated, systhreads)
    shell.cmdloop()

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


def unpack_configuration(section):
    config = configparser.ConfigParser()
    config.read("./source/config.ini")
    items = config[section]

    out = {
        "bitsize": items.getint("bitsize"),
        "gain_center": items.getfloat("gain_center"),
        "gain_sigma": items.getfloat("gain_sigma"),
        "offset_center": items.getfloat("offset_center"),
        "offset_sigma": items.getfloat("offset_sigma"),
        "lightout_center": items.getfloat("lightout_center"),
        "lightout_sigma": items.getfloat("lightout_sigma"),
    }
    return out


def get_from(fitspath: Path, console, use_cache=True):
    console.log(":question_mark: Looking for data..")
    cached = upaths.CACHEDIR().joinpath(fitspath.name).with_suffix(".pkl.gz")
    if cached.is_file() and use_cache:
        out = pd.read_pickle(cached)
        console.log(
            "[bold red]:exclamation_mark: Data were loaded from cache."
        )
    elif fitspath.is_file():
        out = pandas_from_LV0d5(fitspath)
        console.log(":open_book: Data loaded.")
        if use_cache:
            # save data to cache
            # TODO: saving to cache needs a dedicated function
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
    filtered_percentual = (
        100 * (events_pre_filter - len(data)) / events_pre_filter
    )
    console.log(
        ":white_check_mark: Filtered out {:.1f}% of the events.".format(
            filtered_percentual
        )
    )
    return data


def warn_about_flagged(flagged, channels, console):
    interface.sections_rule(
        console, "[bold italic]Warning", style="red", align="center"
    )

    num_flagged = len(
        set([item for sublist in flagged.values() for item in sublist])
    )
    num_channels = len(
        [item for sublist in channels.values() for item in sublist]
    )
    message = (
        "In total, {} channels out of {} were flagged.\n"
        "For more details, see the log file.".format(num_flagged, num_channels)
    )

    console.print(message)
    return True


# boring stuff hereafter
# noinspection PyTypeChecker
def process_results(calibration, eventlist, filepath, output_format, console):
    xhistograms = calibration.xhistograms
    shistograms = calibration.shistograms
    xradsources = calibration.xradsources()
    sradsources = calibration.sradsources()
    xfit_results = calibration.xfit
    sfit_results = calibration.sfit
    sdd_calibration = calibration.sdd_cal
    effective_louts = calibration.optical_coupling
    write_report = get_writer(output_format)

    if not sdd_calibration and not effective_louts:
        console.log("[bold red]:red_circle: Calibration failed.")
    elif not sdd_calibration or not effective_louts:
        console.log(
            "[bold yellow]:yellow_circle: Calibration partially completed. "
        )
    else:
        console.log(":green_circle: Calibration complete.")

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

    if not eventlist is None:
        draw_and_save_calibrated_spectra(
            eventlist,
            xradsources,
            sradsources,
            upaths.XSPPLOT(filepath),
            upaths.SSPPLOT(filepath),
        )
        console.log(":chart_increasing: Saved calibrated spectra plots.")
    return True


if __name__ == "__main__":
    args = parse_args()
    systhreads = min(8, cpu_count())
    start_log(upaths.LOGFILE(args.filepath))
    run(args)
