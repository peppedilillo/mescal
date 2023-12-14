from time import sleep

from rich.console import Console
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

from source.cli.beaupy.beaupy import select
from source.paths import LOGOPATH
from source.utils import get_version

mescal_text = "[magenta]mescal[/]"

header_text = (
    Text()
    .append(
        "\nWelcome to ",
        style="italic",
    )
    .append(
        Text().from_markup(mescal_text),
    )
    .append(
        ", a software to analyze data from the HERMES payloads.\n",
        style="italic",
    )
    .append(
        "Made with <3 by the HERMES-TP/SP calibration team. Since 2021.\n",
        style="italic",
    )
    .append(
        "Software version: {}".format(get_version()),
        style="italic",
    )
)


def logo():
    with open(LOGOPATH, "r") as logofile:
        as_string = logofile.read()
    return as_string


def hello():
    console = Console(theme=Theme({"log.time": "cyan"}))
    for i, line in enumerate(logo().split("\n")):
        console.print(
            line,
            style="bold color({})".format(int(i + 160)),
            justify="center",
        )
        sleep(0.1)
    console.print(header_text, justify="center")
    return console


def sections_rule(console, *args, **kwargs):
    sleep(0.2)
    console.print()
    console.rule(*args, **kwargs)
    console.print()


def logcal_rule(console):
    sections_rule(console, "[bold italic]Calibration log[/]", style="green")


def warning_rule(console):
    sections_rule(console, ":eyes: [bold italic]Warning[/] :eyes:", style="red")


def shell_rule(console):
    sections_rule(console, "[bold italic]Shell[/]", style="green")


def shell_section_header(console, header):
    width = int(console.width / 2)
    space = " " * max(int((width - len(header)) / 2), 0)
    message = space + "%s" % str(header)

    renderable = ["\n%s" % str(message)]
    doc_rule = [Rule(style="green")]
    renderable += doc_rule
    return renderable


class small_section:
    def __init__(self, console, header="", message=""):
        self.console = console
        self.width = int(self.console.width / 2)
        self.header = header
        self.message = message

    def __enter__(self):
        if self.header:
            space = " " * max((self.width - len(self.header)) // 2, 0)
            padded_header = space + "%s" % str(self.header)
            self.print("\n[i]%s[/i]" % str(padded_header))
            self.print(Rule(style="green"))
        if self.message:
            self.print("[i]" + self.message + "\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.console.print("")
        # from fluent python
        if exc_type is ZeroDivisionError:
            print("Please DO NOT divide by zero!")
            return True

    def print(self, *args, **kwargs):
        self.console.print(*args, **kwargs, width=self.width)


# Define custom progress bar
def progress_bar(console):
    out = Progress(
        SpinnerColumn(),
        TextColumn("[i]Working..[/] [progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("-"),
        TimeElapsedColumn(),
        TextColumn("-"),
        TimeRemainingColumn(),
        console=console,
        expand=True,
        transient=True,
    )
    return out


if __name__ == "__main__":
    # Use custom progress bar
    console_ = Console()
    with small_section(
        console=console_, header="Example", message="This is a small section"
    ) as s:
        s.print("some text")

    with progress_bar(console_) as p:
        for i in p.track(range(200)):
            # Do something here
            sleep(0.1)
