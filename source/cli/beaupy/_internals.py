"""
Copyright (c) 2022 Peter Vyboch
Copyright (c) 2018 Hans Schülein

Code from userpetereon@github's Beaupy (thanks).
https://github.com/petereon/beaupy
A copy of the license file has been attached in the `licenses' folder.

Introduced small changes to allow passing of console object and redistribution.

~Peppe

----

A Python library of interactive CLI elements you have been looking for

"""

from contextlib import contextmanager
from typing import Iterator, List, Union

from rich._emoji_replace import _emoji_replace
from rich.console import Console
from rich.console import ConsoleRenderable
from rich.live import Live
from rich.text import Text


class ValidationError(Exception):
    pass


class ConversionError(Exception):
    pass


def _replace_emojis(text: str) -> str:
    return str(_emoji_replace(text, "  "))


def _format_option_select(
    i: int, cursor_index: int, option: str, cursor_style: str, cursor: str
) -> str:
    return "{}{}".format(
        (
            f"[{cursor_style}]{cursor}[/{cursor_style}] "
            if i == cursor_index
            else " " * (len(_replace_emojis(cursor)) + 1)
        ),
        option,
    )


def _render_option_select_multiple(
    option: str,
    ticked: bool,
    tick_character: str,
    tick_style: str,
    selected: bool,
    cursor_style: str,
) -> str:
    prefix = "\[{}]".format(
        " " * len(_replace_emojis(tick_character))
    )  # noqa: W605
    if ticked:
        prefix = (
            f"\[[{tick_style}]{tick_character}[/{tick_style}]]"  # noqa: W605
        )
    if selected:
        option = f"[{cursor_style}]{option}[/{cursor_style}]"
    return f"{prefix} {option}"


def _update_rendered(
    live: Live, renderable: Union[ConsoleRenderable, str]
) -> None:
    live.update(renderable=renderable)
    live.refresh()


def _render_prompt(
    secure: bool,
    typed_values: List[str],
    prompt: str,
    cursor_position: int,
    error: str,
) -> str:
    render_value = (
        len(typed_values) * "*" if secure else "".join(typed_values)
    ) + " "
    render_value = Text(render_value)
    render_value.stylize("black on white", cursor_position, cursor_position + 1)
    confirm_text = Text("\n\n(Confirm with enter, exit with esc)")
    confirm_text.stylize("bold", 16, 21)
    render_value = Text.from_markup(prompt + "\n") + render_value + confirm_text
    if error:
        render_value = f"{render_value}\n[red]Error:[/red] {error}"
    return render_value


@contextmanager
def _cursor_hidden(console: Console) -> Iterator:
    console.show_cursor(False)
    yield
    console.show_cursor(True)
