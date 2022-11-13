import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from source import paths
from source.calibration import PEAKS_PARAMS
from source.io import (
    write_eventlist_to_fits,
    write_report_to_excel,
)
from source.plot import (
    _uncalibrated,
    draw_and_save_channels_sspectra,
    draw_and_save_channels_xspectra,
    draw_and_save_diagns,
    draw_and_save_lins,
    draw_and_save_uncalibrated,
)

import string, sys
from rich.rule import Rule


class Cmd:
    """
    This class belongs to python's standard library. See All rights reserved.
    For more info, see: https://docs.python.org/3/library/cmd.html.
    The code was adapter to use the rich's console protocol in place of
     stdin and stdout.

    ~Peppe

    --------------------------
    A simple framework for writing line-oriented command interpreters.

    These are often useful for test harnesses, administrative tools, and
    prototypes that will later be wrapped in a more sophisticated interface.

    A Cmd instance or subclass instance is a line-oriented interpreter
    framework.  There is no good reason to instantiate Cmd itself; rather,
    it's useful as a superclass of an interpreter class you define yourself
    in order to inherit Cmd's methods and encapsulate action methods.

    """

    prompt =  "(Cmd) "
    identchars = string.ascii_letters + string.digits + "_"
    ruler = "-"
    lastcmd = ""
    intro = None
    doc_leader = ""
    doc_header = "Documented commands (type help <topic>):"
    misc_header = "Miscellaneous help topics:"
    undoc_header = "Undocumented commands:"
    nohelp = "*** No help on %s"
    use_rawinput = 1

    def __init__(self, console, completekey="tab", stdin=None):
        """Instantiate a line-oriented interpreter framework.

        The optional argument 'completekey' is the readline name of a
        completion key; it defaults to the Tab key. If completekey is
        not None and the readline module is available, command completion
        is done automatically. The optional argument stdin specify alternate
        input file objects; if not specified, sys.stdin is used.

        """
        if stdin is not None:
            self.stdin = stdin
        else:
            self.stdin = sys.stdin
        self.console = console
        self.cmdqueue = []
        self.completekey = completekey

    def cmdloop(self, intro=None):
        """Repeatedly issue a prompt, accept input, parse an initial prefix
        off the received input, and dispatch to action methods, passing them
        the remainder of the line as argument.

        """

        self.preloop()
        if self.use_rawinput and self.completekey:
            try:
                import readline

                self.old_completer = readline.get_completer()
                readline.set_completer(self.complete)
                readline.parse_and_bind(self.completekey + ": complete")
            except ImportError:
                pass
        try:
            if intro is not None:
                self.intro = intro
            if self.intro:
                self.console.print(self.intro)
            stop = None
            while not stop:
                if self.cmdqueue:
                    line = self.cmdqueue.pop(0)
                else:
                    if self.use_rawinput:
                        try:
                            line = input(self.prompt)
                        except EOFError:
                            line = "EOF"
                    else:
                        self.console.print(self.prompt)
                        line = self.stdin.readline()
                        if not len(line):
                            line = "EOF"
                        else:
                            line = line.rstrip("\r\n")
                line = self.precmd(line)
                stop = self.onecmd(line)
                stop = self.postcmd(stop, line)
            self.postloop()
        finally:
            if self.use_rawinput and self.completekey:
                try:
                    import readline

                    readline.set_completer(self.old_completer)
                except ImportError:
                    pass

    def precmd(self, line):
        """Hook method executed just before the command line is
        interpreted, but after the input prompt is generated and issued.

        """
        return line

    def postcmd(self, stop, line):
        """Hook method executed just after a command dispatch is finished."""
        return stop

    def preloop(self):
        """Hook method executed once when the cmdloop() method is called."""
        pass

    def postloop(self):
        """Hook method executed once when the cmdloop() method is about to
        return.

        """
        pass

    def parseline(self, line):
        """Parse the line into a command name and a string containing
        the arguments.  Returns a tuple containing (command, args, line).
        'command' and 'args' may be None if the line couldn't be parsed.
        """
        line = line.strip()
        if not line:
            return None, None, line
        elif line[0] == "?":
            line = "help " + line[1:]
        elif line[0] == "!":
            if hasattr(self, "do_shell"):
                line = "shell " + line[1:]
            else:
                return None, None, line
        i, n = 0, len(line)
        while i < n and line[i] in self.identchars:
            i = i + 1
        cmd, arg = line[:i], line[i:].strip()
        return cmd, arg, line

    def onecmd(self, line):
        """Interpret the argument as though it had been typed in response
        to the prompt.

        This may be overridden, but should not normally need to be;
        see the precmd() and postcmd() methods for useful execution hooks.
        The return value is a flag indicating whether interpretation of
        commands by the interpreter should stop.

        """
        cmd, arg, line = self.parseline(line)
        if not line:
            return self.emptyline()
        if cmd is None:
            return self.default(line)
        self.lastcmd = line
        if line == "EOF":
            self.lastcmd = ""
        if cmd == "":
            return self.default(line)
        else:
            try:
                func = getattr(self, "do_" + cmd)
            except AttributeError:
                return self.default(line)
            return func(arg)

    def emptyline(self):
        """Called when an empty line is entered in response to the prompt.

        If this method is not overridden, it repeats the last nonempty
        command entered.

        """
        if self.lastcmd:
            return self.onecmd(self.lastcmd)

    def default(self, line):
        """Called on an input line when the command prefix is not recognized.

        If this method is not overridden, it prints an error message and
        returns.

        """
        self.console.print("*** Unknown syntax: %s" % line)

    def completedefault(self, *ignored):
        """Method called to complete an input line when no command-specific
        complete_*() method is available.

        By default, it returns an empty list.

        """
        return []

    def completenames(self, text, *ignored):
        dotext = "do_" + text
        return [a[3:] for a in self.get_names() if a.startswith(dotext)]

    def complete(self, text, state):
        """Return the next possible completion for 'text'.

        If a command has not been entered, then complete against command list.
        Otherwise try to call complete_<command> to get list of completions.
        """
        if state == 0:
            import readline

            origline = readline.get_line_buffer()
            line = origline.lstrip()
            stripped = len(origline) - len(line)
            begidx = readline.get_begidx() - stripped
            endidx = readline.get_endidx() - stripped
            if begidx > 0:
                cmd, args, foo = self.parseline(line)
                if cmd == "":
                    compfunc = self.completedefault
                else:
                    try:
                        compfunc = getattr(self, "complete_" + cmd)
                    except AttributeError:
                        compfunc = self.completedefault
            else:
                compfunc = self.completenames
            self.completion_matches = compfunc(text, line, begidx, endidx)
        try:
            return self.completion_matches[state]
        except IndexError:
            return None

    def get_names(self):
        # This method used to pull in base class attributes
        # at a time dir() didn't do it yet.
        return dir(self.__class__)

    def complete_help(self, *args):
        commands = set(self.completenames(*args))
        topics = set(a[5:] for a in self.get_names() if a.startswith("help_" + args[0]))
        return list(commands | topics)

    def do_help(self, arg):
        'List available commands with "help" or detailed help with "help cmd".'
        if arg:
            # XXX check arg syntax
            try:
                func = getattr(self, "help_" + arg)
            except AttributeError:
                try:
                    doc = getattr(self, "do_" + arg).__doc__
                    if doc:
                        self.console.print("%s" % str(doc))
                        return
                except AttributeError:
                    pass
                self.console.print("%s\n" % str(self.nohelp % (arg,)))
                return
            func()
        else:
            names = self.get_names()
            cmds_doc = []
            cmds_undoc = []
            topics = set()
            for name in names:
                if name[:5] == "help_":
                    topics.add(name[5:])
            names.sort()
            # There can be duplicates if routines overridden
            prevname = ""
            for name in names:
                if name[:3] == "do_":
                    if name == prevname:
                        continue
                    prevname = name
                    cmd = name[3:]
                    if cmd in topics:
                        cmds_doc.append(cmd)
                        topics.remove(cmd)
                    elif getattr(self, name).__doc__:
                        cmds_doc.append(cmd)
                    else:
                        cmds_undoc.append(cmd)
            # self.console.print("%s"%str(self.doc_leader))
            self.print_topics(self.doc_header, cmds_doc)
            self.print_topics(self.misc_header, sorted(topics))
            self.print_topics(self.undoc_header, cmds_undoc)

    def print_topics(self, header, cmds):
        if cmds:
            columns = int(self.console.width / 2)
            space = " " * max(int((columns - len(header)) / 2), 0)
            message = space + "%s" % str(header)
            self.console.print("\n%s" % str(message))
            if self.ruler:
                doc_rule = Rule()
                self.console.print(doc_rule, width=columns)
            self.columnize(cmds, columns - 1)

    def columnize(self, list, displaywidth=80):
        """Display a list of strings as a compact set of columns.

        Each column is only as wide as necessary.
        Columns are separated by two spaces (one was not legible enough).
        """
        if not list:
            self.console.print("<empty>\n")
            return

        nonstrings = [i for i in range(len(list)) if not isinstance(list[i], str)]
        if nonstrings:
            raise TypeError(
                "list[i] not a string for i in %s" % ", ".join(map(str, nonstrings))
            )
        size = len(list)
        if size == 1:
            self.console.print("%s\n" % str(list[0]))
            return
        # Try every row count from 1 upwards
        for nrows in range(1, len(list)):
            ncols = (size + nrows - 1) // nrows
            colwidths = []
            totwidth = -2
            for col in range(ncols):
                colwidth = 0
                for row in range(nrows):
                    i = row + nrows * col
                    if i >= size:
                        break
                    x = list[i]
                    colwidth = max(colwidth, len(x))
                colwidths.append(colwidth)
                totwidth += colwidth + 2
                if totwidth > displaywidth:
                    break
            if totwidth <= displaywidth:
                break
        else:
            nrows = len(list)
            ncols = 1
            colwidths = [0]
        for row in range(nrows):
            texts = []
            cmds = []
            for col in range(ncols):
                i = row + nrows * col
                if i >= size:
                    x = ""
                else:
                    x = list[i]
                    texts.append(x)
                    cmds.append(x)
            while texts and not texts[-1]:
                del texts[-1]
            for col in range(len(texts)):
                texts[col] = texts[col].ljust(colwidths[col])
            # markup in red unavailable commands
            for i, (text, cmd) in enumerate(zip(texts, cmds)):
                if not self.can(cmd):
                    texts[i] = "[red]" + text + "[/]"
            self.console.print("%s" % str("  ".join(texts)))
        self.console.print()

    def can(self, x):
        if "can_" + x not in dir(self.__class__):
            return True
        else:
            func = getattr(self, "can_" + x)
            return func()


INVALID_ENTRY = 0


def parse_chns(arg):
    quadrants = ['A', 'B', 'C', 'D']
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
        print("Invalid entry.\n"
        "Channel ID not in standard form.\n" 
        "(e.g., d04, A31, B02)"
        )
        return INVALID_ENTRY


def parse_limits(arg):
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
        print("Invalid entry.\n"
        "Entry must be two, different positive integers.\n"
        "Largest integers must follow the smallest.\n"
        "Example: '19800 20100'."
        )
        return INVALID_ENTRY


class MescalShell(Cmd):
    intro = (
        "This is [bold purple]mescal[/] shell. "
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

    def do_set_xfit_lims(self, arg):
        """Reset channel X peaks position."""
        parsed_arg = parse_chns(arg)
        if parsed_arg is INVALID_ENTRY:
            return False

        quad, ch = parsed_arg
        for source, decay in self.calibration.xradsources().items():
            arg = self.console.input(source + ": ")
            parsed_arg = parse_limits(arg)
            if parsed_arg is INVALID_ENTRY:
                return False
            elif parsed_arg is None:
                continue
            else:
                lim_lo, lim_hi = parsed_arg
                label_lo, label_hi = PEAKS_PARAMS
                self.calibration.xpeaks[quad].loc[ch, (source, label_lo)] = int(lim_lo)
                self.calibration.xpeaks[quad].loc[ch, (source, label_hi)] = int(lim_hi)

    def do_retry(self, arg):
        """Launches calibration again."""
        with self.console.status(self.spinner_message):
            self.calibration._run_calibration()

    def do_plot(self, arg):
        """Plots uncalibrated data from a channel."""
        parsed_arg = parse_chns(arg)
        if parsed_arg is not None:
            quad, ch = parsed_arg
            fig, ax = _uncalibrated(
                self.calibration.xhistograms.bins,
                self.calibration.xhistograms.counts[quad][ch],
                self.calibration.shistograms.bins,
                self.calibration.shistograms.counts[quad][ch],
                figsize=(9, 4.5),
            )
            ax.set_title("Uncalibrated plot - CH{:02d}Q{}".format(ch, quad))
            plt.show(block=False)

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

    def can_save_speak_table(self):
        if self.calibration.speaks:
            return True
        return False

    def do_save_speak_table(self, arg):
        """Save X peak tables."""
        if not self.can_save_speak_table():
            self.console.print(self.failure)
            return False
        with self.console.status(self.spinner_message):
            write_report_to_excel(
                self.calibration.speaks, paths.XFTREPORT(self.filename),
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
