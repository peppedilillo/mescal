from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich.progress import track
from rich.pretty import pprint
from time import sleep

asciilogo = \
'''
       __ __   ____   ___    __  ___   ____   ____       _____         __
      / // /  / __/  / _ \  /  |/  /  / __/  / __/ ____ / ___/ ___ _  / /
     / _  /  / _/   / , _/ / /|_/ /  / _/   _\ \  /___// /__  / _ `/ / / 
    /_//_/  /___/  /_/|_| /_/  /_/  /___/  /___/       \___/  \_,_/ /_/  
'''


def boot():
    console = Console()
    for i, line in enumerate(asciilogo.split("\n", )):
        console.print(line, style="color({})".format(i + 160));
        sleep(0.1)
    print("")
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


def prettyprint(x):
    return pprint(x)


def shutdown(console):
    console.print("\nShutting down, goodbye!\n")
    return
