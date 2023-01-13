import argparse
import atexit
import configparser
import logging
import sys
from collections import namedtuple
from os import cpu_count
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from source import paths
from source.calibrate import PEAKS_PARAMS, Calibrate
from source.cli import interface
from source.cli.beaupy.beaupy import select_multiple
from source.cli.interface import logo, sections_rule
from source.cmd import Cmd
from source.detectors import Detector
from source.io import (
    get_writer,
    pandas_from_LV0d5,
    write_eventlist_to_fits,
    write_report_to_excel,
)
from source.plot import (
    draw_and_save_calibrated_spectra,
    draw_and_save_channels_sspectra,
    draw_and_save_channels_xspectra,
    draw_and_save_diagns,
    draw_and_save_lins,
    draw_and_save_mapcounts,
    draw_and_save_mapres,
    draw_and_save_qlooks,
    draw_and_save_slo,
    draw_and_save_uncalibrated,
    mapcounts,
    uncalibrated,
)
from source.radsources import radsources_dicts
from source.utils import get_version

description = (
    "A script to automatically calibrate HERMES-TP/SP "
    "acquisitions of known radioactive sources."
)
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "model",
    choices=["dm", "fm1", "pfm", "fm2", "fm3"],
    help="hermes flight model to calibrate. ",
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
    "--adc",
    choices=["LYRA-BE", "CAEN-DT5740"],
    default="LYRA-BE",
    help="select which adc configuration to use." "defaults to LYRA-BE.",
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


def start_logger(args):
    """
    Starts logger in default output folder and logs user command line arguments.
    """
    logfile = paths.LOGFILE(args.filepath)
    with open(logfile, "w") as f:
        f.write(logo())
        len_logo = len(logo().split("\n")[0])
        version_message = "version " + get_version()
        if len_logo > len(version_message) + 1:
            f.write(
                " " * (len_logo - len(version_message) + 1) + version_message + "\n\n"
            )
        else:
            f.write(version_message + "\n\n")
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="[%(funcName)s() @ %(filename)s (L%(lineno)s)] "
        "%(levelname)s: %(message)s",
    )

    message = "parser arguments = " + str(args)[10:-1]
    logging.info(message)
    return True


@atexit.register
def end_log():
    """
    Kills loggers on shutdown.
    """
    logging.shutdown()


def check_system():
    """
    Perfoms system inspection to choose matplotlib backend
    and threads number (max 4).
    """
    if sys.platform.startswith("win") or sys.platform.startswith("linux"):
        if "TkAgg" in matplotlib.rcsetup.all_backends:
            pass  # matplotlib.use("TkAgg")
    elif sys.platform.startswith("mac"):
        if "macosx" in matplotlib.rcsetup.all_backends:
            matplotlib.use("macosx")
    systhreads = min(4, cpu_count())

    logging.info("detected {} os".format(sys.platform))
    logging.info("using matplotlib backend {}".format(matplotlib.get_backend()))
    logging.info("running over {} threads".format(systhreads))
    return systhreads


def parse_args():
    """
    user arguments parsing.
    """
    args = parser.parse_args()
    args.filepath = Path(args.filepath)
    args.radsources = args.radsources.upper().split(",")
    if (args.model is None) and args.temperature:
        parser.error(
            "if a temperature arguments is specified "
            "a model argument must be specified too "
        )
    return args


def unpack_configuration(adc):
    """
    unpacks ini configuration file into a dict.
    """
    config = configparser.ConfigParser()
    config.read("./source/config.ini")
    general = config["general"]
    adcitems = config[adc]
    out = {
        "xpeaks_mincounts": general.getint("xpeaks_mincounts"),
        "filter_retrigger": general.getfloat("filter_retrigger"),
        "filter_spurious": general.getboolean("filter_spurious"),
        "binning": adcitems.getint("binning"),
        "gain_center": adcitems.getfloat("gain_center"),
        "gain_sigma": adcitems.getfloat("gain_sigma"),
        "offset_center": adcitems.getfloat("offset_center"),
        "offset_sigma": adcitems.getfloat("offset_sigma"),
        "lightout_center": adcitems.getfloat("lightout_center"),
        "lightout_sigma": adcitems.getfloat("lightout_sigma"),
    }

    message = "config.ini parameters = " + str(out)[1:-1]
    logging.info(message)
    return out


# TODO: refactoring. caching should have its own function.
def get_from(fitspath: Path, console, use_cache=True):
    """
    deals with data load and caching.
    """
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
            out.to_pickle(cached)
            console.log(":blue_book: Data saved to cache.")
    else:
        raise FileNotFoundError("could not find input datafile.")
    return out


INVALID_ENTRY = 0


class Mescal(Cmd):
    """
    Main program class implementing calibration workflow and shell loop.
    """

    intro = "Type help or ? for a list of commands.\n"
    prompt = "[cyan]\[mescalSH] "
    spinner_message = "Working.."
    unknown_command_message = (
        "[red]Unknown command.[/]\nType help or ? for a list of commands."
    )
    invalid_command_message = (
        "[red]Command unavailable.[/]\nCannnot execute with present calibration."
    )
    invalid_channel_message = (
        "[red]Invalid channel.[/]\n"
        "Channel ID not in standard form (e.g., d04, A30, B02)."
    )
    invalid_limits_message = (
        "[red]Invalid limits.[/]\n"
        "Entry must be two different sorted integers (e.g., 19800 20100)."
    )

    def __init__(self, args, threads):
        console = interface.hello()
        super().__init__(console)
        self.args = args
        self.config = unpack_configuration(args.adc)
        self.threads = threads

        sections_rule(console, "[bold italic]Calibration log", style="green")
        with console.status("Initializing.."):
            data = get_from(self.args.filepath, self.console, self.args.cache)
            radsources = radsources_dicts(self.args.radsources)
            detector = Detector(self.args.model)

        with console.status("Calibrating.."):
            self.calibration = Calibrate(
                detector,
                radsources,
                configuration=self.config,
                console=self.console,
                nthreads=self.threads,
            )
            self.eventlist = self.calibration(data)

        with console.status("Processing results.."):
            self._process_results(self.args.fmt, self.args.filepath)

        if any(self.calibration.flagged):
            sections_rule(console, "[bold italic]Warning", style="red")
            self._warn_about_flagged()

        sections_rule(console, "[bold italic]Shell", style="green")
        self.cmdloop()

    def _warn_about_flagged(self):
        """Tells user about channels for which calibration
        could not be completed.
        """
        sublists = self.calibration.flagged.values()
        num_flagged = len(set([item for sublist in sublists for item in sublist]))
        num_channels = len([ch for quad, chs in self.calibration.channels.items() for ch in chs])
        message = (
            "In total, {} channels out of {} were flagged.\n"
            "For more details, see the log file.".format(num_flagged, num_channels)
        )

        self.console.print(message)
        return True

    def _process_results(self, output_format, filepath):
        """Prepares and exports base calibration results."""
        write_report = get_writer(output_format)

        if not self.calibration.sdd_cal and not self.calibration.optical_coupling:
            self.console.log("[bold red]:red_circle: Calibration failed.")
        elif not self.calibration.sdd_cal or not self.calibration.optical_coupling:
            self.console.log(
                "[bold yellow]:yellow_circle: Calibration partially complete. "
            )
        else:
            self.console.log(":green_circle: Calibration complete.")

        if self.calibration.sdd_cal:
            write_report(
                self.calibration.sdd_cal, path=paths.CALREPORT(filepath),
            )
            self.console.log(":blue_book: Wrote SDD calibration results.")
            write_report(
                self.calibration.en_res, path=paths.RESREPORT(filepath),
            )
            self.console.log(":blue_book: Wrote energy resolution results.")
            draw_and_save_qlooks(
                self.calibration.sdd_cal,
                path=paths.QLKPLOT(filepath),
                nthreads=self.threads,
            )
            self.console.log(":chart_increasing: Saved X fit quicklook plots.")

        if self.calibration.optical_coupling:
            write_report(
                self.calibration.optical_coupling, path=paths.SLOREPORT(filepath),
            )
            self.console.log(":blue_book: Wrote scintillators calibration results.")
            draw_and_save_slo(
                self.calibration.optical_coupling,
                path=paths.SLOPLOT(filepath),
                nthreads=self.threads,
            )
            self.console.log(":chart_increasing: Saved light-output plots.")

        if (self.eventlist is not None) and (not self.eventlist.empty):
            draw_and_save_calibrated_spectra(
                self.eventlist,
                self.calibration.xradsources(),
                self.calibration.sradsources(),
                paths.XSPPLOT(filepath),
                paths.SSPPLOT(filepath),
            )
            self.console.log(":chart_increasing: Saved calibrated spectra plots.")
        return True

    # shell prompt commands
    def do_quit(self, arg):
        """Quits mescal.
        It's the only do-command to return True.
        """
        self.console.print("Ciao! :wave:\n")
        return True

    def do_retry(self, arg):
        """Launches calibration again."""
        with self.console.status(self.spinner_message):
            sections_rule(self.console, "[bold italic]Calibration log", style="green")
            self.calibration._calibrate()
            self._process_results(self.args.fmt, self.args.filepath)
            sections_rule(self.console, "[bold italic]Shell", style="green")
        return False

    def do_setxlim(self, arg):
        """Reset channel X peaks position for user selected channels."""
        parsed_arg = parse_chns(arg)
        if parsed_arg is INVALID_ENTRY:
            self.console.print(self.invalid_channel_message)
            return False

        quad, ch = parsed_arg
        for source, decay in self.calibration.xradsources().items():
            arg = self.console.input(source + ": ")
            parsed_arg = parse_limits(arg)
            if parsed_arg is INVALID_ENTRY:
                self.console.print(self.invalid_limits_message)
                return False
            elif parsed_arg is None:
                continue
            else:
                lim_lo, lim_hi = parsed_arg
                label_lo, label_hi = PEAKS_PARAMS
                self.calibration.xpeaks[quad].loc[ch, (source, label_lo)] = int(lim_lo)
                self.calibration.xpeaks[quad].loc[ch, (source, label_hi)] = int(lim_hi)

        message = "reset xfit limits for channel {}{}".format(quad, ch)
        logging.info(message)
        return False

    def do_plothist(self, arg):
        """Plots uncalibrated data from a channel."""
        parsed_arg = parse_chns(arg)
        if parsed_arg is INVALID_ENTRY:
            self.console.print(self.invalid_channel_message)
            return False

        quad, ch = parsed_arg
        fig, ax = uncalibrated(
            self.calibration.xhistograms.bins,
            self.calibration.xhistograms.counts[quad][ch],
            self.calibration.shistograms.bins,
            self.calibration.shistograms.counts[quad][ch],
        )
        ax.set_title("Uncalibrated plot - CH{:02d}Q{}".format(ch, quad))
        plt.show(block=False)
        return False

    def do_mapcounts(self, arg):
        """Plots a map of counts per-channel."""
        fig, ax = mapcounts(self.calibration.counts(), self.calibration.detector.map)
        plt.show(block=False)
        return False

    def do_export(self, arg):
        """Prompts user on optional exports."""
        cmds = [
            "svhistplot",
            "svdiags",
            "svtabfit",
            "svxplots",
            "svlinplots",
            "svsplots",
            "svmapres",
            "svmapcounts",
        ]

        Option = namedtuple("Option", ["label", "command", "ticked",])
        all_options = [
            Option("Uncalibrated histogram plots", "svhistplot", True),
            Option("X diagnostic plots", "svxdiags", True),
            Option("S diagnostic plots", "svsdiags", False),
            Option("Linearity plots", "svlinplots", False),
            Option("Per-channel X spectra plots", "svxplots", False),
            Option("Per-channel S spectra plots", "svsplots", False),
            Option("Energy resolution map", "svmapres", True),
            Option("Channel counts map", "svmapcounts", True),
            Option("Fit tables", "svtabfit", True),
            Option("Save calibrated events to fits file", "svevents", False),
        ]

        options = [o for o in all_options if self.can(o.command)]
        options_labels = [o.label for o in options]
        options_commands = [o.command for o in options]
        options_ticked = [i for i, v in enumerate(options) if v.ticked]

        prompt_user_with_menu = True
        if prompt_user_with_menu:
            selection = select_multiple(
                options_labels,
                self.console,
                ticked_indices=options_ticked,
                return_indices=True,
            )
        else:
            # do them all
            selection = [i for i, _ in enumerate(options)]

        for cmd in [options_commands[i] for i in selection]:
            do = getattr(self, cmd)
            do("")
        return False

    def can(self, x):
        """Checks if a command can be executed."""
        if "can_" + x not in dir(self.__class__):
            # if a can_ method is not defined we assume the command
            # to always be executable.
            return True
        else:
            func = getattr(self, "can_" + x)
            return func()

    def svmapcounts(self, arg):
        """Saves a map of per-channel counts."""
        with self.console.status(self.spinner_message):
            draw_and_save_mapcounts(
                self.calibration.counts(),
                self.calibration.detector.map,
                paths.CNTPLOT(self.args.filepath),
            )
        return False

    def svhistplot(self, arg):
        """Save raw acquisition histogram plots."""
        with self.console.status(self.spinner_message):
            draw_and_save_uncalibrated(
                self.calibration.xhistograms,
                self.calibration.shistograms,
                paths.UNCPLOT(self.args.filepath),
                nthreads=self.threads,
            )
        return False

    def can_svmapres(self):
        if self.calibration.xradsources().keys() and self.calibration.en_res:
            return True
        return False

    def svmapres(self, arg):
        """Saves a map of channels' energy resolution."""
        if not self.can_svmapres():
            self.console.print(self.invalid_command_message)
            return False
        with self.console.status(self.spinner_message):
            decays = self.calibration.xradsources()
            source = sorted(decays, key=lambda source: decays[source].energy)[0]
            draw_and_save_mapres(
                source,
                self.calibration.en_res,
                self.calibration.detector.map,
                paths.RESPLOT(self.args.filepath),
            )
        return False

    def can_svxdiags(self):
        if self.calibration.xfit:
            return True
        return False

    def svxdiags(self, arg):
        """Save X peak detection diagnostics plots."""
        if not self.can_svxdiags():
            self.console.print(self.invalid_command_message)
            return False
        with self.console.status(self.spinner_message):
            if self.calibration.xfit:
                draw_and_save_diagns(
                    self.calibration.xhistograms,
                    self.calibration.xfit,
                    paths.XDNPLOT(self.args.filepath),
                    nthreads=self.threads,
                )
        return False

    def can_svsdiags(self):
        if self.calibration.sfit:
            return True
        return False

    def svsdiags(self, arg):
        """Save X peak detection diagnostics plots."""
        if not self.can_svsdiags():
            self.console.print(self.invalid_command_message)
            return False
        with self.console.status(self.spinner_message):
            if self.calibration.sfit:
                draw_and_save_diagns(
                    self.calibration.shistograms,
                    self.calibration.sfit,
                    paths.SDNPLOT(self.args.filepath),
                    nthreads=self.threads,
                )
        return False

    def can_svtabfit(self):
        if self.calibration.xfit or self.calibration.sfit:
            return True
        return False

    def svtabfit(self, arg):
        """Save X fit tables."""
        if not self.can_svtabfit():
            self.console.print(self.invalid_command_message)
            return False
        with self.console.status(self.spinner_message):
            if self.calibration.xfit:
                write_report_to_excel(
                    self.calibration.xfit, paths.XFTREPORT(self.args.filepath),
                )
            if self.calibration.sfit:
                write_report_to_excel(
                    self.calibration.sfit, paths.SFTREPORT(self.args.filepath),
                )
        return False

    def can_svxplots(self):
        if self.calibration.sdd_cal:
            return True
        return False

    def svxplots(self, arg):
        """Save calibrated X channel spectra."""
        if not self.can_svxplots():
            self.console.print(self.invalid_command_message)
            return False
        with self.console.status(self.spinner_message):
            draw_and_save_channels_xspectra(
                self.calibration.xhistograms,
                self.calibration.sdd_cal,
                self.calibration.xradsources(),
                paths.XCSPLOT(self.args.filepath),
                nthreads=self.threads,
            )
        return False

    def can_svlinplots(self):
        if self.calibration.sdd_cal:
            return True
        return False

    def svlinplots(self, arg):
        """Save SDD linearity plots."""
        if not self.can_svlinplots():
            self.console.print(self.invalid_command_message)
            return False
        with self.console.status(self.spinner_message):
            draw_and_save_lins(
                self.calibration.sdd_cal,
                self.calibration.xfit,
                self.calibration.xradsources(),
                paths.LINPLOT(self.args.filepath),
                nthreads=self.threads,
            )
        return False

    def can_svsplots(self):
        if self.calibration.optical_coupling:
            return True
        return False

    def svsplots(self, arg):
        """Save calibrated gamma channel spectra."""
        if not self.can_svsplots():
            self.console.print(self.invalid_command_message)
            return False
        with self.console.status(self.spinner_message):
            draw_and_save_channels_sspectra(
                self.calibration.shistograms,
                self.calibration.sdd_cal,
                self.calibration.optical_coupling,
                self.calibration.sradsources(),
                paths.SCSPLOT(self.args.filepath),
                nthreads=self.threads,
            )
        return False

    def can_svevents(self):
        if self.eventlist is not None:
            return True
        return False

    def svevents(self, arg):
        """Save calibrated events to fits file."""
        if not self.can_svevents():
            self.console.print(self.invalid_command_message)
            return False
        with self.console.status(self.spinner_message):
            write_eventlist_to_fits(
                self.eventlist, paths.EVLFITS(self.args.filepath),
            )
        return False


def parse_chns(arg):
    """
    Shell helper.
    """
    quadrants = ["A", "B", "C", "D"]
    chn_strings = ["{0:02d}".format(i) for i in range(32)]
    stripped_arg = arg.strip()
    if (
        arg
        and (arg[0].upper() in quadrants)
        and (arg[1:3] in chn_strings)
        and len(stripped_arg) == 3
    ):
        quad = arg[0]
        ch = int(arg[1:3])
        return quad, ch
    else:
        return INVALID_ENTRY


def parse_limits(arg):
    """
    Shell helper.
    """
    if arg == "":
        return None

    arglist = arg.strip().split(" ")
    if (
        len(arglist) == 2
        and arglist[0].isdigit()
        and arglist[1].isdigit()
        and int(arglist[0]) < int(arglist[1])
    ):
        botlim = int(arglist[0])
        toplim = int(arglist[1])
        return botlim, toplim
    else:
        return INVALID_ENTRY


if __name__ == "__main__":
    user_arguments = parse_args()
    start_logger(user_arguments)
    systhreads = check_system()
    Mescal(user_arguments, systhreads)
