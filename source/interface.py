from time import sleep

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm
from rich.pretty import pprint
from rich.prompt import IntPrompt

from source.upaths import LOGOPATH


hdr_text = Text()\
    .append("Welcome to ", style='italic')\
    .append("mescal", style='purple bold')\
    .append(", a software to analyze HERMES-TP/SP data.\n", style='italic')


def hello():
    console = Console()
    with open(LOGOPATH, 'r') as logo:
        for i, line in enumerate(logo.readlines()):
            console.print(line.strip("\n"), style="bold color({})".format(int(i + 160)), justify='center');
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


def flagged_message(flagged, onchannels):
    message = Text("\nWhile processing data I've found {} channels out of {} "
                   "for which calibration could not be completed."
                   .format(sum(len(v) for v in flagged.values()), sum(len(v) for v in onchannels.values())))
    return message


def prompt_user_flagged():
    message = Text("Display flagged channels? ", style='italic')
    return Confirm.ask(message)


def prettyprint(x, **kwargs):
    return pprint(x, **kwargs)


def shutdown(console):
    console.print("\nShutting down, goodbye! :waving_hand:\n")
    return


def options_message(options):
    line_end = (lambda i: '\n')
    message = Text.assemble("\nAnything else?\n\n",
                            *(Text.assemble(("\t{:2d}. ".format(i),"bold magenta"),
                                            option.display + line_end(i)) for i, option in enumerate(options)))
    return message


def prompt_user_about(options):
    message = Text("Select:")
    choices = [*range(len(options))]
    return options[IntPrompt.ask(message, choices=[str(i) for i in choices])]


def print_rule(console, *args, **kwargs):
    sleep(0.2)
    console.print()
    console.rule(*args, **kwargs)
