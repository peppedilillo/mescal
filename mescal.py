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
import source.errors as err
from source.calibrate import PEAKS_PARAMS, Calibrate, ImportedCalibration
from source.checks import check_results
from source.cli import elementsui as ui
from source.cli.beaupy.beaupy import prompt, select, select_multiple
from source.cli.cmd import Cmd
from source.detectors import supported_models
from source.eventlist import preprocess, perchannel_counts
from source.io import pandas_from_LV0d5, read_sdd_calibration_report, read_lightout_report
from source.plot import mapcounts, mapenres, uncalibrated, spectrum_xs, histogram
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

    # fmt off
    intro = "Type help or ? for a list of commands.\n"
    prompt = "[cyan]\[mescalSH] "
    spinner_message = "Working.."
    unknown_command_message = (
        "[red]Unknown command.[/]\n" "[i]Type help or ? for a list of commands.[/i]\n"
    )
    invalid_command_message = "[red]Command unavailable.[/]\n"
    invalid_channel_message = (
        "[red]Invalid channel.[/]\n"
        "[i]Channel ID must be in standard form "
        "(e.g., d04, A30, B02).[/i]\n"
    )
    no_counts_message = "[red]No events observed for this channel.[/]\n"
    invalid_limits_message = (
        "[red]Invalid limits.[/]\n"
        "[i]Entries must be two, different, "
        "sorted integers (e.g., 19800 20100).[/i]\n"
    )
    invalid_table_message = (
        "[red]Invalid table.[/]\n"
        "[i]Make sure the table you are providing has the right columns.[/i]"
    )
    invalid_format_message = (
        "[red]The file appears to be in wrong format.[/]\n"
        "[i]The command `loadcal` expects .xslx table format.[/i]"
    )
    # fmt on

    def __init__(self):
        console = ui.hello()
        super().__init__(console)
        self.args = parser.parse_args()
        self.filepath = self.get_filepath()
        self.model = self.get_model()
        self.radsources = self.get_radsources()
        self.start_logger()
        self.config = self.unpack_configuration()
        self.threads = self.check_system()
        self.data = None
        self.waste = None
        self.calibrations = {}
        self.calibration = None

        ui.logcal_rule(self.console)
        with console.status("Initializing.."):
            raw_data = self.fetch_data()
            self.data, self.waste = preprocess(
                raw_data,
                model=self.model,
                filter_spurious=self.config["filter_spurious"],
                filter_retrigger=self.config["filter_retrigger"],
                console=self.console,
            )

        with console.status("Analyzing data.."):
            calibration = Calibrate(
                self.model,
                self.radsources,
                configuration=self.config,
                console=self.console,
                nthreads=self.threads,
            )
            self.register(calibration)
            self.calibration = calibration
            self.calibration(self.data)

        with console.status("Processing results.."):
            self.print_calibration_status()
            self.export_essentials()

        failed_tests = check_results(
            self.calibration,
            self.data,
            self.waste,
            self.config,
        )
        if failed_tests:
            ui.warning_rule(self.console)
            self.display_warning(failed_tests)

        ui.shell_rule(self.console)
        self.cmdloop()

    def register(self, calibration):
        label = "calib-{}".format(len(self.calibrations))
        self.calibrations[label] = calibration
        logging.info("added calibration {} to register.".format(label))
        return calibration

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
        message = (
            "[italic]Which file are you calibrating?\n"
            "[yellow]Hint: You can drag & drop.[/yellow]"
            "[/italic]\n"
        )
        message_error = (
            "[italic][red]The file you entered does not exists.[/red]\n"
            "Which file are you calibrating?\n"
            "[yellow]Hint: You can drag & drop.[/yellow]\n"
            "[/italic]\n"
        )
        if self.args.filepath is not None:
            return Path(self.args.filepath)
        filepath = prompt_user_on_filepath(message, message_error, self.console)
        if filepath is None:
            self.console.print("So soon? Ciao :wave:!\n")
            exit()
        return filepath

    def get_model(self):
        def prompt_user_on_model():
            # fmt: off
            cursor_index = ([0] + [
                i for i, d in enumerate(supported_models())
                if d in self.filepath.name
            ])[-1]
            # fmt: on
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
            return self.args.model
        model = prompt_user_on_model()
        return model

    def get_radsources(self):
        def prompt_user_on_radsources():
            message = (
                "[italic]With which radioactive sources?\n"
                "[yellow]Hint: Pressing esc or selecting no source will cause mescal to skip "
                "the calibration process. You will still be able to visualize data.[/yellow]\n"
                "[/italic]\n"
            )
            legend = (
                "\n\n"
                "(mark=[bold]space[/bold], "
                "confirm=[bold]enter[/bold], "
                "cancel=[bold]skip[/bold])"
            )
            radsources = select_multiple(
                options=supported_sources(),
                tick_character=":radioactive:",
                ticked_indices=list(range(len(supported_sources()))),
                console=self.console,
                intro=message,
                transient=True,
                legend=legend
            )
            return radsources

        if self.args.source is not None:
            return self.args.source
        radsources = prompt_user_on_radsources()
        return radsources

    def display_warning(self, failed_tests):
        """Tells user about channels for which calibration
        could not be completed.
        """
        if "flagged_channels" in failed_tests:
            sublists = self.calibration.flagged.values()
            num_flagged = len(set([item for sublist in sublists for item in sublist]))
            num_channels = len(
                [ch for quad, chs in self.calibration.channels.items() for ch in chs]
            )
            message = (
                "[i][yellow]"
                "I was unable to complete calibration for {} channels out of {}."
                "[/yellow]\n"
                "For more details, see the log file.".format(num_flagged, num_channels)
            )
            self.console.print(message)
        if "too_many_filtered_events" in failed_tests:
            message = (
                "[i][yellow]"
                "A significant fraction was filtered away."
                "[/yellow]\n"
                "Check filter parameters in 'config.ini'."
            )
            self.console.print(message)
        if "filter_retrigger_off" in failed_tests:
            message = (
                "[i][yellow]"
                "Retrigger filter is off."
                "[/yellow]\n"
                "You can enable it through 'config.ini'."
            )
            self.console.print(message)
        if "filter_spurious_off" in failed_tests:
            message = (
                "[i][yellow]"
                "Spurious events filter is off."
                "[/yellow]\n"
                "You can enable it through 'config.ini'."
            )
            self.console.print(message)
        if "time_outliers" in failed_tests:
            message = (
                "[i][yellow]"
                "Found large outliers in your time data."
                "[/yellow]\n"
                "These events will not be displayed through 'timehist' command."
            )
            self.console.print(message)
        return True

    def print_calibration_status(self):
        """Prepares and exports base calibration results."""
        if not self.radsources:
            return
        if not self.calibration.sdd_calibration and not self.calibration.lightoutput:
            self.console.log("[bold red]:red_circle: Calibration failed.")
        elif not self.calibration.sdd_calibration or not self.calibration.lightoutput:
            self.console.log(
                "[bold yellow]:yellow_circle: Calibration partially complete. "
            )
        else:
            self.console.log(":green_circle: Calibration complete.")

    def export_essentials(self):
        exporter = self.calibration.get_exporter(self.filepath, self.args.fmt)

        if exporter.can__write_sdd_calibration_report:
            exporter.write_sdd_calibration_report()
            self.console.log(":blue_book: Wrote SDD calibration results.")
        if exporter.can__write_energy_res_report:
            exporter.write_energy_res_report()
            self.console.log(":blue_book: Wrote energy resolution results.")
        if exporter.can__draw_qlooks_sdd:
            exporter.draw_qlooks_sdd()
            self.console.log(":chart_increasing: Saved X fit quicklook plots.")
        if exporter.can__write_lightoutput_report:
            exporter.write_lightoutput_report()
            self.console.log(":blue_book: Wrote light output results.")
        if exporter.can__draw_qlook_scint:
            exporter.draw_qlook_scint()
            self.console.log(":chart_increasing: Saved light output plots.")
        if exporter.can__draw_spectrum:
            exporter.draw_spectrum()
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

    def can_plotcal(self, arg):
        if self.calibration.eventlist is not None:
            return True

    def do_plotcal(self, arg):
        fig, axs = spectrum_xs(
            self.calibration.eventlist,
            self.calibration.xradsources(),
            self.calibration.sradsources(),
        )
        plt.show(block=False)
        return False

    def can_plotraw(self, arg):
        return True

    def do_plotraw(self, arg):
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
        ax.set_title("Uncalibrated plot channel {}{:02d}".format(quad, ch))
        plt.show(block=False)
        return False

    def can_timehist(self, arg):
        return True

    def do_timehist(self, arg):
        """Plots a histogram of counts in time for selected channel."""
        parsed_arg = parse_chns(arg)
        if parsed_arg is INVALID_ENTRY:
            self.console.print(self.invalid_channel_message)
            return False

        quad, ch = parsed_arg
        binning = 1.0
        try:
            counts, bins = self.calibration.timehist(quad, ch, binning)
        except err.BadDataError:
            counts, bins = self.calibration.timehist(
                quad, ch, binning, neglect_outliers=True
            )

        fig, ax = histogram(
            counts,
            bins[:-1],
        )
        ax.set_title(
            "Count in time over channel {}{:02d}, binning {} s".format(
                quad, ch, binning
            )
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Counts")
        plt.show(block=False)
        return False

    def can_retry(self, arg):
        if self.radsources and isinstance(self.calibration, ImportedCalibration):
            return True
        return False

    def do_retry(self, arg):
        """Launches calibration again."""
        with self.console.status(self.spinner_message):
            ui.logcal_rule(self.console)
            self.calibration._calibrate()
            self.export_essentials()
            ui.shell_rule(self.console)
        return False

    def can_setlim(self, arg):
        if self.radsources and isinstance(self.calibration, ImportedCalibration):
            return True
        return False

    def do_setlim(self, arg):
        """Reset channel X peaks position for user selected channels."""
        parsed_arg = parse_chns(arg)
        if parsed_arg is INVALID_ENTRY:
            self.console.print(self.invalid_channel_message)
            return False
        quad, ch = parsed_arg
        if (quad not in self.calibration.channels) or (
            ch not in self.calibration.channels[quad]
        ):
            self.console.print(self.no_counts_message)
            return False

        for source, decay in self.calibration.xradsources().items():
            arg = input(source + ": ")
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

        for source, decay in self.calibration.sradsources().items():
            arg = input(source + ": ")
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

        message = "reset fit limits for channel {}{:02d}".format(quad, ch)
        logging.info(message)
        return False

    def can_mapcount(self, arg):
        if self.calibration.data is not None:
            return True
        return False

    def do_mapcount(self, arg):
        """Plots a map of counts per-channel."""
        counts = self.calibration.count()
        fig, ax = mapcounts(
            counts,
            self.calibration.detector.map,
            title="Per-channel events count map",
        )
        plt.show(block=False)
        return False

    def can_mapbad(self, arg):
        if self.waste is not None:
            return True
        return False

    def do_mapbad(self, arg):
        """Plots a map of counts per-channel from filtered data."""
        counts = perchannel_counts(self.waste, self.calibration.channels, key="all")
        fig, ax = mapcounts(
            counts,
            self.calibration.detector.map,
            cmap="binary_u",
            title="Per-channel filtered events count map",
        )
        plt.show(block=False)
        return False

    def can_mapres(self, arg):
        if self.calibration.xradsources().keys() and self.calibration.resolution:
            return True
        return False

    def do_mapres(self, arg):
        """Saves a map of channels' energy resolution."""
        decays = self.calibration.xradsources()
        source = sorted(decays, key=lambda source: decays[source].energy)[0]
        fig, ax = mapenres(
            source,
            self.calibration.resolution,
            self.calibration.detector.map,
        )
        plt.show(block=False)
        return False

    def can_loadcal(self, arg):
        return True

    def do_loadcal(self, arg):
        """Loads and existing calibration."""
        message_sdd = (
            "[italic]Enter path for sdd calibration file.\n"
            "[yellow]Hint: You can drag & drop.[/yellow]"
            "[/italic]\n"
        )
        message_lout = (
            "[italic]Enter path for light output calibration file.\n"
            "[yellow]Hint: You can drag & drop.[/yellow]"
            "[/italic]\n"
        )
        message_error = (
            "[italic][red]The file you entered does not exists.[/red]\n"
            "Which file are you calibrating?\n"
            "[yellow]Hint: You can drag & drop.[/yellow]\n"
            "[/italic]\n"
        )
        answer = prompt_user_on_filepath(message_sdd, message_error, self.console)
        if answer is None:
            return False
        try:
            sddcal = read_sdd_calibration_report(answer)
        except err.WrongTableError:
            self.console.print(self.invalid_table_message)
            return False
        except err.FormatNotSupportedError:
            self.console.print(self.invalid_format_message)
            return False

        answer = prompt_user_on_filepath(message_lout, message_error, self.console)
        if answer is None:
            return False
        try:
            local = read_lightout_report(answer)
        except err.WrongTableError:
            self.console.print(self.invalid_table_message)
            return False
        except err.FormatNotSupportedError:
            self.console.print(self.invalid_format_message)
            return False

        with self.console.status(self.spinner_message):
            ui.logcal_rule(self.console)
            try:
                newcal = ImportedCalibration(
                    self.model,
                    self.config,
                    sdd_calibration=sddcal,
                    lightoutput=local,
                    console=self.console,
                    nthreads=self.threads,
                )
            except ValueError:
                self.console.print(
                    "The file doesn't appear to be in a valid format.\n"
                    "Presently we only support .xlsx table."
                )
                return False
            self.register(newcal)
            self.calibration = newcal
            self.calibration(self.data)
            ui.shell_rule(self.console)
        return False

    def can_swapcal(self, arg):
        if self.calibrations:
            return True
        return False

    def do_swapcal(self, arg):
        calib_label = select(
            options=list(self.calibrations.keys()),
            cursor=":flying_saucer:",
            console=self.console,
            intro="[italic]Select one calibration.[/italic]\n\n",
        )
        if calib_label is None:
            return False
        self.calibration = self.calibrations[calib_label]
        return False

    def can_export(self, arg):
        return True

    def do_export(self, arg):
        """Prompts user on optional data product exports."""
        exporter = self.calibration.get_exporter(self.filepath, self.args.fmt)

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
                [exporter.draw_rawspectra],
                [exporter.can__draw_rawspectra],
                True,
            ),
            Option(
                "diagnostic plots",
                [
                    exporter.draw_xdiagnostic,
                    exporter.draw_sdiagnostics,
                ],
                [
                    exporter.can__draw_xdiagnostic,
                    exporter.can__draw_sdiagnostics,
                ],
                True,
            ),
            Option(
                "linearity plots",
                [exporter.draw_linearity],
                [exporter.can__draw_linearity],
                False,
            ),
            Option(
                "spectra plots per channel",
                [
                    exporter.draw_sspectra,
                    exporter.draw_xspectra,
                ],
                [
                    exporter.can__draw_sspectra,
                    exporter.can__draw_xspectra,
                ],
                False,
            ),
            Option(
                "maps",
                [
                    exporter.draw_map_counts,
                    exporter.draw_map_resolution,
                ],
                [
                    exporter.can__draw_map_counts,
                    exporter.can__draw_map_resolution,
                ],
                True,
            ),
            Option(
                "fit tables",
                [
                    exporter.write_xfit_report,
                    exporter.write_sfit_report,
                ],
                [
                    exporter.can__write_xfit_report,
                    exporter.can__write_sfit_report,
                ],
                True,
            ),
            Option(
                "calibrated events fits",
                [exporter.write_eventlist],
                [exporter.can__write_eventlist],
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


def prompt_user_on_filepath(message, message_error, console):
    filepath = None
    text = message
    while filepath is None:
        answer = prompt(
            text,
            console=console,
            target_type=str,
        )
        if answer is None:
            return None
        if not answer:
            continue
        answer = answer.strip()
        if not Path(answer).exists():
            text = message_error
        else:
            filepath = Path(answer)
    return filepath


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
