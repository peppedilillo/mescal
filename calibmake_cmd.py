import argparse
import atexit
import logging
import configparser
import sys
from collections import namedtuple
from os import cpu_count
from time import sleep
from pathlib import Path
import cmd
import pandas as pd

from source import cmd
from source import interface, paths
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

RETRIGGER_TIME_IN_S = 20 * (10 ** -6)

description = (
    "A script to automatically calibrate HERMES-TP/SP "
    "acquisitions of known radioactive sources."
)
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "model",
    choices=["fm1", "pfm", "fm2"],
    help="hermes flight model to calibrate. " "supported models: fm1, pfm, fm2.",
)
parser.add_argument(
    "radsources",
    help="radioactive sources used for calibration. "
    "separated by comma, e.g. `Fe,Cd,Cs`."
    "currently supported sources: Fe, Cd, Am, Cs.",
)
parser.add_argument(
    "filepath", help="input acquisition file in standard 0.5 fits format.",
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
    spinner_message = "Working.."

    def __init__(self, console, filename, config, calibration, threads):
        super().__init__(console)
        self.calibration = calibration
        self.filename = filename
        self.threads = threads
        self.config = config

    def can_save_rawhist_plots(self):
        return True

    def do_save_rawhist_plots(self, arg):
        """Save raw acquisition histogram plots."""
        with self.console.status(self.spinner_message):
            draw_and_save_uncalibrated(
                self.calibration.xhistograms,
                self.calibration.shistograms,
                paths.UNCPLOT(self.filename),
                self.threads,
            )

    def can_save_xdiagns_plots(self):
        if self.calibration.xfit:
            return True
        return False

    def do_save_xdiagns_plots(self, arg):
        """Save X peak detection diagnostics plots."""
        if not self.can_save_xdiagns_plots():
            self.console.print(self.failure)
            return False
        with self.console.status(self.spinner_message):
            draw_and_save_diagns(
                self.calibration.xhistograms,
                self.calibration.xfit,
                paths.XDNPLOT(self.filename),
                self.config["margin_diag_plot"],
                self.threads,
            )

    def can_save_xfit_table(self):
        if self.calibration.xfit:
            return True
        return False

    def do_save_xfit_table(self, arg):
        """Save X fit tables."""
        if not self.can_save_xfit_table():
            self.console.print(self.failure)
            return False
        with self.console.status(self.spinner_message):
            write_report_to_excel(
                self.calibration.xfit, paths.XFTREPORT(self.filename),
            )

    def can_save_xspectra_plots(self):
        if self.calibration.sdd_cal:
            return True
        return False

    def do_save_xspectra_plots(self, arg):
        """Save X sources fit tables."""
        if not self.can_save_xspectra_plots():
            self.console.print(self.failure)
            return False
        with self.console.status(self.spinner_message):
            draw_and_save_channels_xspectra(
                self.calibration.xhistograms,
                self.calibration.sdd_cal,
                self.calibration.xradsources(),
                paths.XCSPLOT(self.filename),
                self.threads,
            )

    def can_save_xlin_plots(self):
        if self.calibration.sdd_cal:
            return True
        return False

    def do_save_xlin_plots(self, args):
        """Save SDD linearity plots."""
        if not self.can_save_xlin_plots():
            self.console.print(self.failure)
            return False
        with self.console.status(self.spinner_message):
            draw_and_save_lins(
                self.calibration.sdd_cal,
                self.calibration.xfit,
                self.calibration.xradsources(),
                paths.LINPLOT(self.filename),
                self.threads,
            )

    def can_save_sdiagns_plots(self):
        if self.calibration.sfit:
            return True
        return False

    def do_save_sdiagns_plots(self, args):
        """Save S peak detection diagnostics plots."""
        if not self.can_save_sdiagns_plots():
            self.console.print(self.failure)
            return False
        with self.console.status(self.spinner_message):
            draw_and_save_diagns(
                self.calibration.shistograms,
                self.calibration.sfit,
                paths.SDNPLOT(self.filename),
                self.config["margin_diag_plot"],
                self.threads,
            )

    def can_save_sfit_table(self):
        if self.calibration.sfit:
            return True
        return False

    def do_save_sfit_table(self, arg):
        """Save gamma sources fit tables."""
        if not self.can_save_sfit_table():
            self.console.print(self.failure)
            return False
        with self.console.status(self.spinner_message):
            write_report_to_excel(
                self.calibration.sfit, paths.SFTREPORT(self.filename),
            )

    def can_save_sspectra_plots(self):
        if self.calibration.optical_coupling:
            return True
        return False

    def do_save_sspectra_plots(self, arg):
        """Save gamma sources fit tables."""
        if not self.can_save_sspectra_plots():
            self.console.print(self.failure)
            return False
        with self.console.status(self.spinner_message):
            draw_and_save_channels_sspectra(
                self.calibration.shistograms,
                self.calibration.sdd_cal,
                self.calibration.optical_coupling,
                self.calibration.sradsources(),
                paths.SCSPLOT(self.filename),
                self.threads,
            )

    def can_save_event_fits(self):
        if self.calibration.eventlist is not None:
            return True
        return False

    def do_save_event_fits(self, arg):
        """Save calibrated events to fits file."""
        if not self.can_save_event_fits():
            self.console.print(self.failure)
            return False
        with self.console.status(self.spinner_message):
            write_eventlist_to_fits(
                self.calibration.eventlist, paths.EVLFITS(self.filename),
            )

    def do_all(self, arg):
        """Executes every executable command."""
        cmds = [cmd[4:] for cmd in dir(self.__class__) if cmd[:4] == "can_"]
        for cmd in cmds:
            if self.can(cmd):
                do = getattr(self, "do_" + cmd)
                do("")

    def do_quit(self, arg):
        """Quit mescal."""
        self.console.print("Ciao! :wave:\n")
        return True


def run(args):
    console = interface.hello()
    config_dict = unpack_configuration(args.configuration)

    sections_rule(console, "[bold italic]Calibration log", style="green")
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
        process_results(calibrated, eventlist, args.filepath, args.fmt, console)

    if any(calibrated.flagged):
        sections_rule(console, "[bold italic]Warning", style="red")
        warn_about_flagged(calibrated.flagged, channels, console)

    sections_rule(console, "[bold italic]Shell", style="green")
    shell = MescalShell(console, args.filepath, config_dict, calibrated, systhreads)
    shell.cmdloop()
    return True


def sections_rule(console, *args, **kwargs):
    sleep(0.2)
    console.print()
    console.rule(*args, **kwargs)
    console.print()


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
        "margin_diag_plot": items.getint("margin_diag_plot"),
    }
    return out


def get_from(fitspath: Path, console, use_cache=True):
    console.log(":question_mark: Looking for data..")
    cached = paths.CACHEDIR().joinpath(fitspath.name).with_suffix(".pkl.gz")
    if cached.is_file() and use_cache:
        out = pd.read_pickle(cached)
        console.log("[bold yellow]:yellow_circle: Data were loaded from cache.")
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
    filtered_percentual = 100 * (events_pre_filter - len(data)) / events_pre_filter
    console.log(
        ":white_check_mark: Filtered out {:.1f}% of the events.".format(
            filtered_percentual
        )
    )
    return data


def warn_about_flagged(flagged, channels, console):
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
    write_report = get_writer(output_format)

    if not calibration.sdd_cal and not calibration.optical_coupling:
        console.log("[bold red]:red_circle: Calibration failed.")
    elif not calibration.sdd_cal or not calibration.optical_coupling:
        console.log("[bold yellow]:yellow_circle: Calibration partially completed. ")
    else:
        console.log(":green_circle: Calibration complete.")

    if calibration.sdd_cal:
        write_report(
            calibration.sdd_cal, path=paths.CALREPORT(filepath),
        )
        console.log(":blue_book: Wrote SDD calibration results.")

        draw_and_save_qlooks(
            calibration.sdd_cal, path=paths.QLKPLOT(filepath), nthreads=systhreads,
        )
        console.log(":chart_increasing: Saved X fit quicklook plots.")

    if calibration.optical_coupling:
        write_report(
            calibration.optical_coupling, path=paths.SLOREPORT(filepath),
        )
        console.log(":blue_book: Wrote scintillators calibration results.")
        draw_and_save_slo(
            calibration.optical_coupling,
            path=paths.SLOPLOT(filepath),
            nthreads=systhreads,
        )
        console.log(":chart_increasing: Saved light-output plots.")

    if eventlist is not None:
        draw_and_save_calibrated_spectra(
            eventlist,
            calibration.xradsources(),
            calibration.sradsources(),
            paths.XSPPLOT(filepath),
            paths.SSPPLOT(filepath),
        )
        console.log(":chart_increasing: Saved calibrated spectra plots.")
    return True


if __name__ == "__main__":
    args = parse_args()
    systhreads = min(8, cpu_count())
    start_log(paths.LOGFILE(args.filepath))
    run(args)
