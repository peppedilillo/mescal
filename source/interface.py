from time import sleep

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm
from rich.progress import track
from rich.pretty import pprint

from source.upaths import LOGOPATH


hdr_text = Text()\
    .append("Welcome to ", style='italic')\
    .append("mescal v1.0", style='purple bold')\
    .append(", a software toolkit to analyze HERMES-TP/SP data.\n", style='italic')


def boot():
    console = Console()
    with open(LOGOPATH, 'r') as logo:
        for i, line in enumerate(logo.readlines()):
            console.print(line.strip("\n"), style="bold color({})".format(int(i + 160)), justify='center');
            sleep(0.1)
    console.print()
    console.rule(hdr_text)
    console.print()
    return console


def progress_bar(onchannels, log_to):
    return {asic: track(onchannels[asic], "Processing ASIC {}..".format(asic), console=log_to)
            for asic in onchannels.keys()}


def df_to_table(df, title):
    table = Table(title=title)
    for i, col in enumerate(df.columns):
        table.add_column(col, justify="right", style="cyan", no_wrap=True)
    for index, row in df.iterrows():
        table.add_row(*map((lambda x: "{:.2f}".format(x)), row.values))
    return table


def confirm_prompt(message):
    return Confirm.ask(message)


def prettyprint(x, **kwargs):
    return pprint(x, **kwargs)


def shutdown(console):
    console.print("\nShutting down, goodbye!\n")
    return
