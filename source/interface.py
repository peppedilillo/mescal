from rich.console import Console
from rich.table import Table
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
    for line in asciilogo.split("\n"):
        console.print(line); sleep(0.1)
    return console


def shutdown(console):
    console.print("\nShutting down, goodbye! :waving_hand:\n")
    return


def df_to_table(df, title):
    table = Table(title=title)
    for i, col in enumerate(df.columns):
        table.add_column(col, justify="right", style="cyan", no_wrap=True)
    for index, row in df.iterrows():
        table.add_row(*map((lambda x: "{:.2f}".format(x)), row.values))
    return table