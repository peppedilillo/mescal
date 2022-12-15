from time import sleep

from rich.console import Console
from rich.text import Text

from source.paths import LOGOPATH
from source.utils import get_version

hdr_text = (
    Text()
    .append("\nWelcome to ", style="italic")
    .append("mescal", style="bold purple")
    .append(
        ", a software to analyze data from the HERMES payloads.\n",
        style="italic",
    )
    .append(
        "Made with <3 by the HERMES-TP/SP calibration team. Since 2021.\n",
        style="italic",
    )
    .append("Software version: {}".format(get_version()), style="italic")
)


def logo():
    with open(LOGOPATH, "r") as logofile:
        as_string = logofile.read()
    return as_string


def hello():
    console = Console()
    for i, line in enumerate(logo().split("\n")):
        console.print(
            line,
            style="bold color({})".format(int(i + 160)),
            justify="center",
        )
        sleep(0.1)
    console.print(hdr_text, justify="center")
    return console


def sections_rule(console, *args, **kwargs):
    sleep(0.2)
    console.print()
    console.rule(*args, **kwargs)
    console.print()
