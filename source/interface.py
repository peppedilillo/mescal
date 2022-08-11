from time import sleep
from datetime import datetime

from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text

from source.upaths import LOGOPATH

hdr_text = (
    Text()
    .append("Welcome to ", style="italic")
    .append("mescal", style="purple bold")
    .append(", a software to analyze HERMES-TP/SP data.\n", style="italic")
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
    print_rule(console, hdr_text)
    return console


def df_to_table(df, title):
    table = Table(title=title)
    for i, col in enumerate(df.columns):
        table.add_column(col, justify="right", style="cyan", no_wrap=True)
    for index, row in df.iterrows():
        table.add_row(*map((lambda x: "{:.2f}".format(x)), row.values))
    return table


def shutdown(console):
    console.print("Shutting down, goodbye! :waving_hand:\n")
    return


def options_message(options):
    line_end = lambda i: "\n"
    message = Text.assemble(
        "Anything else?\n\n",
        *(
            Text.assemble(
                ("\t{:2d}. ".format(i), "bold magenta"), option.display + line_end(i)
            )
            for i, option in enumerate(options)
        )
    )
    return message


def prompt_user_about(options):
    message = Text("Select:")
    choices = [*range(len(options))]
    return options[IntPrompt.ask(message, choices=[str(i) for i in choices])]


def print_rule(console, *args, **kwargs):
    sleep(0.2)
    console.print()
    console.rule(*args, **kwargs)
    console.print()
