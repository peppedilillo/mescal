import string
import sys

from rich.columns import Columns
from rich.rule import Rule

from source.cli.elementsui import small_section


class Cmd:
    """
    This class belongs to python's standard library.
    For more info, see: https://docs.python.org/3/library/cmd.html.
    The code was adapted to use the rich's console protocol in place of
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

    prompt = "(Cmd) "
    identchars = string.ascii_letters + string.digits + "_"
    ruler = "-"
    lastcmd = ""
    intro = None
    doc_leader = ""
    doc_header = "Documented commands"
    misc_header = "Miscellaneous help topics:"
    undoc_header = "Undocumented commands:"
    unknown_command_message = "Unknown command."
    nohelp = "*** No help on %s"

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

    def ansi_prompt(self):
        """turns a markup prompt to ansi string"""
        with self.console.capture() as capture:
            self.console.print(self.prompt, end="")
        str_output = capture.get()
        return str_output

    def cmdloop(self, intro=None):
        """Repeatedly issue a prompt, accept input, parse an initial prefix
        off the received input, and dispatch to action methods, passing them
        the remainder of the line as argument."""

        self.preloop()

        if self.completekey:
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
                    try:
                        line = input(self.ansi_prompt())
                    except EOFError:
                        line = "EOF"

                line = self.precmd(line)
                stop = self.onecmd(line)
                stop = self.postcmd(stop, line)
            self.postloop()
        finally:
            if self.completekey:
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
        return False

    def default(self, line):
        """Called on an input line when the command prefix is not recognized.

        If this method is not overridden, it prints an error message and
        returns.

        """
        self.console.print(self.unknown_command_message)

    def completedefault(self, *ignored):
        """Method called to complete an input line when no command-specific
        complete_*() method is available.

        By default, it returns an empty list.

        """
        return []

    def completenames(self, text, *ignored):
        if text == "":
            return []
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
            self.print_topics(self.doc_header, cmds_doc,  message="Type help <topic>.")
            self.print_topics(self.misc_header, sorted(topics))
            self.print_topics(self.undoc_header, cmds_undoc)

    def print_topics(self, header, cmds, message=''):
        if cmds:
            with small_section(self.console, header, message) as ss:
                cols = Columns(cmds, expand=True, padding=(0, 2))
                ss.print(cols)