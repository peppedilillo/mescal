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
from source.cli import elementsui as ui
from source.cli.beaupy.beaupy import prompt, select, select_multiple
from source.cli.cmd import Cmd
from source.detectors import supported_models
from source.io import Exporter, pandas_from_LV0d5
from source.plot import mapcounts, mapenres, uncalibrated
from source.radsources import supported_sources
from source.utils import get_version

parser = argparse.ArgumentParser()

parser.add_argument(
    "--filepath",
    default=None,
    help="input acquisition file in standard 0.5 fits format.\n"
    "prompt user by default.",
)

parser.add_argument(
    "--model",
    default=None,
    choices=supported_models(),
    help="hermes flight model to calibrate.\n" "prompt user by default.",
)

parser.add_argument(
    "--source",
    default=None,
    action="append",
    choices=supported_sources(),
    help="radioactive sources used for calibration.\n" "prompt user by default.",
)

parser.add_argument(
    "--adc",
    choices=["LYRA-BE", "CAEN-DT5740"],
    default="LYRA-BE",
    help="select which adc configuration to use.\n" "defaults to LYRA-BE.",
)

parser.add_argument(
    "--fmt",
    default="xslx",
    choices=["xslx", "csv", "fits"],
    help="set output format for calibration tables.\n" "defaults to xslx.",
)

parser.add_argument(
    "--cache",
    default=False,
    action="store_true",
    help="enables loading and saving from cache.\n",
)


@atexit.register
def end_logger():
    """
    Kills loggers on shutdown.
    """
    logging.shutdown()


INVALID_ENTRY = 0


class Mescal(Cmd):
    """
    A script implementing calibration workflow and a shell loop.
    """

    intro = "Type help or ? for a list of commands.\n"
    prompt = "[cyan]\[mescalSH] "
    spinner_message = "Working.."
    unknown_command_message = (
        "[red]Unknown command.[/]\n" "[i]Type help or ? for a list of commands.[/i]\n"
    )
    invalid_command_message = "[red]Command unavailable.[/]\n"
    invalid_channel_message = (
        "[red]Invalid channel.[/]\n"
        "[i]Channel ID must be in standard form (e.g., d04, A30, B02).[/i]\n"
    )
    no_counts_message = (
        "[red]No photon events were observed for this channel.[/]\n"
    )
    invalid_limits_message = (
        "[red]Invalid limits.[/]\n"
        "[i]Entries must be two, different, sorted integers (e.g., 19800 20100).[/i]\n"
    )

    def __init__(self):
        console = ui.hello()
        super().__init__(console)
        self.args = self.parse_args()
        self.filepath = self.get_filepath()
        self.model = self.get_model()
        self.radsources = self.get_radsources()
        self.start_logger()
        self.config = self.unpack_configuration()
        self.threads = self.check_system()

        ui.sections_rule(console, "[bold italic]Calibration log", style="green")
        with console.status("Initializing.."):
            data = self.fetch_data()

        with console.status("Calibrating.."):
            self.calibration = Calibrate(
                self.model,
                self.radsources,
                configuration=self.config,
                console=self.console,
                nthreads=self.threads,
            )
            self.calibration(data)

        with console.status("Processing results.."):
            self.exporter = Exporter(
                self.calibration,
                self.filepath,
                self.args.fmt,
                nthreads=self.threads,
            )
            self.print_calibration_status()
            self.export_essentials(self.filepath)

        if any(self.calibration.flagged):
            ui.sections_rule(console, "[bold italic]Warning", style="red")
            self.warn_about_flagged()

        ui.sections_rule(console, "[bold italic]Shell", style="green")
        self.cmdloop()

    def start_logger(self):
        """
        Starts logger in default output folder and logs user command line arguments.
        """
        logfile = paths.LOGFILE(self.filepath)
        with open(logfile, "w") as f:
            f.write(ui.logo())
            len_logo = len(ui.logo().split("\n")[0])
            version_message = "version " + get_version()
            if len_logo > len(version_message) + 1:
                f.write(
                    " " * (len_logo - len(version_message) + 1)
                    + version_message
                    + "\n\n"
                )
            else:
                f.write(version_message + "\n\n")

        # checks that logging was not used before creating the logfile.
        assert len(logging.root.handlers) == 0
        logging.basicConfig(
            filename=logfile,
            level=logging.INFO,
            format="[%(funcName)s() @ %(filename)s (L%(lineno)s)] "
            "%(levelname)s: %(message)s",
        )
        logging.info("user args = {}".format(self.args))
        logging.info("logging calibration for file {}".format(self.filepath))
        logging.info("selected model = {}".format(self.model))
        logging.info("selected sources = {}".format(", ".join(self.radsources)))
        return True

    @staticmethod
    def parse_args():
        args = parser.parse_args()
        return args

    @staticmethod
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

    def unpack_configuration(self):
        """
        unpacks ini configuration file into a dict.
        """
        config = configparser.ConfigParser()
        config.read("./source/config.ini")
        general = config["general"]
        adcitems = config[self.args.adc]
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
    def fetch_data(self, use_cache=False):
        """
        deals with data load and caching.
        """
        self.console.log(":question_mark: Looking for data..")
        cached = paths.CACHEDIR().joinpath(self.filepath.name).with_suffix(".pkl.gz")
        if cached.is_file() and self.args.cache:
            out = pd.read_pickle(cached)
            self.console.log(
                "[bold yellow]:yellow_circle: Data were loaded from cache."
            )
        elif self.filepath.is_file():
            out = pandas_from_LV0d5(self.filepath)
            self.console.log(":open_book: Data loaded.")
            if self.args.cache:
                # save data to cache
                from pickle import DEFAULT_PROTOCOL

                out.to_pickle(cached, protocol=DEFAULT_PROTOCOL)
                self.console.log(":blue_book: Data saved to cache.")
        else:
            raise FileNotFoundError("could not find input datafile.")
        return out

    def get_filepath(self):
        def prompt_user_on_filepath():
            text_default = (
                "[italic]Which file are you calibrating?\n"
                "You can drag & drop.[italic]\n"
            )
            text_error = (
                "[italic][red]The file you entered does not exists.[/red]\n"
                "Which file are you calibrating?\n"
                "You can drag & drop.[italic]\n"
            )
            filepath = None
            text = text_default
            while filepath is None:
                answer = prompt(
                    text,
                    console=self.console,
                    target_type=str,
                )
                if answer == "" or answer is None:
                    continue
                elif not Path(answer.replace(" ", "")).exists():
                    text = text_error
                else:
                    filepath = Path(answer.strip())
            return filepath

        if self.args.filepath is not None:
            filepath = Path(self.args.filepath)
        else:
            filepath = prompt_user_on_filepath()
        return filepath

    def get_model(self):
        def prompt_user_on_model():
            cursor_index = (
                [0]
                + [
                    i
                    for i, d in enumerate(supported_models())
                    if d in self.filepath.name
                ]
            )[-1]
            model = None
            while model is None:
                model = select(
                    options=supported_models(),
                    cursor=":flying_saucer:",
                    cursor_index=cursor_index,
                    console=self.console,
                    intro="[italic]For which model?[/italic]\n\n",
                )
            return model

        if self.args.model is not None:
            model = self.args.model
        else:
            model = prompt_user_on_model()
        return model

    def get_radsources(self):
        def prompt_user_on_radsources():
            radsources = select_multiple(
                options=supported_sources(),
                tick_character=":radioactive:",
                ticked_indices=list(range(len(supported_sources()))),
                console=self.console,
                intro="[italic]With which radioactive sources?[/italic]\n\n",
                transient=True,
            )
            return radsources

        if self.args.source is not None:
            radsources = self.args.source
        else:
            radsources = prompt_user_on_radsources()
        return radsources

    def warn_about_flagged(self):
        """Tells user about channels for which calibration
        could not be completed.
        """
        sublists = self.calibration.flagged.values()
        num_flagged = len(set([item for sublist in sublists for item in sublist]))
        num_channels = len(
            [ch for quad, chs in self.calibration.channels.items() for ch in chs]
        )
        message = (
            "In total, {} channels out of {} were flagged.\n"
            "For more details, see the log file.".format(num_flagged, num_channels)
        )

        self.console.print(message)
        return True

    def print_calibration_status(self):
        """Prepares and exports base calibration results."""
        if not self.radsources:
            return
        if not self.calibration.sdd_cal and not self.calibration.optical_coupling:
            self.console.log("[bold red]:red_circle: Calibration failed.")
        elif not self.calibration.sdd_cal or not self.calibration.optical_coupling:
            self.console.log(
                "[bold yellow]:yellow_circle: Calibration partially complete. "
            )
        else:
            self.console.log(":green_circle: Calibration complete.")

    def export_essentials(self, filepath):
        if self.exporter.can__write_sdd_calibration_report:
            self.exporter.write_sdd_calibration_report()
            self.console.log(":blue_book: Wrote SDD calibration results.")
        if self.exporter.can__write_energy_res_report:
            self.exporter.write_energy_res_report()
            self.console.log(":blue_book: Wrote energy resolution results.")
        if self.exporter.can__draw_qlooks_sdd:
            self.exporter.draw_qlooks_sdd()
            self.console.log(":chart_increasing: Saved X fit quicklook plots.")
        if self.exporter.can__write_scintillator_report:
            self.exporter.write_scintillator_report()
            self.console.log(":blue_book: Wrote scintillators calibration results.")
        if self.exporter.can__draw_qlook_scint:
            self.exporter.draw_qlook_scint()
            self.console.log(":chart_increasing: Saved light-output plots.")
        if self.exporter.can__draw_spectrum:
            self.exporter.draw_spectrum()
            self.console.log(":chart_increasing: Saved calibrated spectra plots.")

    # shell prompt commands
    def can_quit(self, arg):
        return True

    def do_quit(self, arg):
        """Quits mescal.
        It's the only do-command to return True.
        """
        self.console.print("Ciao! :wave:\n")
        return True

    def can_plothist(self, arg):
        return True

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

    def can_retry(self, arg):
        if self.radsources:
            return True
        return False

    def do_retry(self, arg):
        """Launches calibration again."""
        with self.console.status(self.spinner_message):
            ui.sections_rule(
                self.console, "[bold italic]Calibration log", style="green"
            )
            self.calibration._calibrate()
            self.export_essentials(self.filepath)
            ui.sections_rule(self.console, "[bold italic]Shell", style="green")
        return False

    def can_setxlim(self, arg):
        if self.radsources:
            return True
        return False

    def do_setxlim(self, arg):
        """Reset channel X peaks position for user selected channels."""
        parsed_arg = parse_chns(arg)
        if parsed_arg is INVALID_ENTRY:
            self.console.print(self.invalid_channel_message)
            return False
        quad, ch = parsed_arg
        if (quad not in self.calibration.channels) or (ch not in self.calibration.channels[quad]):
            self.console.print(self.no_counts_message)
            return False

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

    def can_setslim(self, arg):
        if self.radsources:
            return True
        return False

    def do_setslim(self, arg):
        """Reset channel S peaks position for user selected channels."""
        parsed_arg = parse_chns(arg)
        if parsed_arg is INVALID_ENTRY:
            self.console.print(self.invalid_channel_message)
            return False
        quad, ch = parsed_arg
        if (quad not in self.calibration.channels) or (ch not in self.calibration.channels[quad]):
            self.console.print(self.no_counts_message)
            return False

        quad, ch = parsed_arg
        for source, decay in self.calibration.sradsources().items():
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
                self.calibration.speaks[quad].loc[ch, (source, label_lo)] = int(lim_lo)
                self.calibration.speaks[quad].loc[ch, (source, label_hi)] = int(lim_hi)

        message = "reset sfit limits for channel {}{}".format(quad, ch)
        logging.info(message)
        return False

    def can_map(self, arg):
        if self.calibration.data is not None:
            return True
        return False

    def do_map(self, arg):
        """Plots a map of counts per-channel."""
        counts = self.calibration.count()
        fig, ax = mapcounts(counts, self.calibration.detector.map)
        plt.show(block=False)
        return False

    def can_mapx(self, arg):
        if self.calibration.eventlist is not None:
            return True
        return False

    def do_mapx(self, arg):
        """Plots a map of counts per-channel."""
        counts = self.calibration.count(key="x")
        fig, ax = mapcounts(counts, self.calibration.detector.map)
        plt.show(block=False)
        return False

    def can_maps(self, arg):
        if self.calibration.eventlist is not None:
            return True
        return False

    def do_maps(self, arg):
        """Plots a map of S counts per-channel."""
        counts = self.calibration.count(key="s")
        fig, ax = mapcounts(counts, self.calibration.detector.map)
        plt.show(block=False)
        return False

    def can_mapbad(self, arg):
        if self.calibration.waste is not None:
            return True
        return False

    def do_mapbad(self, arg):
        """Plots a map of counts per-channel from filtered data."""
        counts = self.calibration.waste_count(key="all")
        fig, ax = mapcounts(counts, self.calibration.detector.map)
        plt.show(block=False)
        return False

    def can_mapres(self, arg):
        if self.calibration.xradsources().keys() and self.calibration.en_res:
            return True
        return False

    def do_mapres(self, arg):
        """Saves a map of channels' energy resolution."""
        decays = self.calibration.xradsources()
        source = sorted(decays, key=lambda source: decays[source].energy)[0]
        fig, ax = mapenres(
            source,
            self.calibration.en_res,
            self.calibration.detector.map,
        )
        plt.show(block=False)
        return False

    def can_export(self, arg):
        return True

    def do_export(self, arg):
        """Prompts user on optional data product exports."""
        Option = namedtuple(
            "Option",
            [
                "label",
                "commands",
                "conditions",
                "ticked",
            ],
        )
        all_options = [
            Option(
                "uncalibrated plots",
                [self.exporter.draw_rawspectra],
                [self.exporter.can__draw_rawspectra],
                True,
            ),
            Option(
                "diagnostic plots",
                [
                    self.exporter.draw_xdiagnostic,
                    self.exporter.draw_sdiagnostics,
                ],
                [
                    self.exporter.can__draw_xdiagnostic,
                    self.exporter.can__draw_sdiagnostics,
                ],
                True,
            ),
            Option(
                "linearity plots",
                [self.exporter.draw_linearity],
                [self.exporter.can__draw_linearity],
                False,
            ),
            Option(
                "spectra plots per channel",
                [
                    self.exporter.draw_sspectra,
                    self.exporter.draw_xspectra,
                ],
                [
                    self.exporter.can__draw_sspectra,
                    self.exporter.can__draw_xspectra,
                ],
                False,
            ),
            Option(
                "maps",
                [
                    self.exporter.draw_map_counts,
                    self.exporter.draw_map_resolution,
                ],
                [
                    self.exporter.can__draw_map_counts,
                    self.exporter.can__draw_map_resolution,
                ],
                True,
            ),
            Option(
                "fit tables",
                [
                    self.exporter.write_xfit_report,
                    self.exporter.write_sfit_report,
                ],
                [
                    self.exporter.can__write_xfit_report,
                    self.exporter.can__write_sfit_report,
                ],
                True,
            ),
            Option(
                "calibrated events fits",
                [self.exporter.write_eventlist],
                [self.exporter.can__write_eventlist],
                False,
            ),
        ]

        options = [o for o in all_options if any(o.conditions)]
        if arg != "all":
            with ui.small_section(self.console, message="Select one or more.") as ss:
                indeces = select_multiple(
                    [o.label for o in options],
                    self.console,
                    ticked_indices=[i for i, v in enumerate(options) if v.ticked],
                    return_indices=True,
                )
        else:
            # do them all
            indeces = [i for i, _ in enumerate(options)]

        selection = [options[i] for i in indeces]
        commands = [c for o in selection for c in o.commands]
        conditions = [c for o in selection for c in o.conditions]
        with ui.progress_bar(self.console) as p:
            for condition, f in p.track(zip(conditions, commands), total=len(commands)):
                if condition:
                    f()
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
    Mescal()
