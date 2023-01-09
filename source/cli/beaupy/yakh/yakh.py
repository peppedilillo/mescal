"""
Forked from petereon's yakh repository (thanks). All rights reserved.
Minimal changes to allow passing of console object and redistribution.

You can find the original repo at https://github.com/petereon/yakh

~Peppe

-----
# yakh

yakh (Yet Another Keypress Handler) tries to handle keypresses from the stdin in
the terminal in high-level platform indendendent manner.

## Usage

```python
from yakh import get_key
from yakh.key import Keys

key = ''
while key not in ['q', Keys.ENTER]:
    key = get_key()
    if key.is_printable:
        print(key)
```

yakh is dead-simple, there is only one function `get_key()` which takes no
arguments and blocks until a key is pressed.

For each keypress it creates an instance of [`Key`](./yakh/key/_key.py#L7)
which holds:

- `.key`: characters representing the keypress
- `.key_codes`: collection of Unicode code point encodings for all the
    characters (given by `ord` function)
- `.is_printable`: printability of the characters in the keypress

Additionally `Key` instances

-  are comparable with another `Key` instances, `str` instances and *Unicode
code point* representations (tuples of integers)
- come with string representation for purposes of printing and string
concatenation, which returns the content of `.key` attribute

## `yakh.key` submodule
`yakh.key` sub-module contains platform dependent representations of certain
keys under `Keys` class. These are available namely for `CTRL` key combinations
and some other common keys.

Full list of keys can be seen [here](./yakh/key/_key.py#L42)
and [here](./yakh/key/_key.py#L81).
"""

import sys
from typing import List

from .key._key import Key

try:
    import fcntl
    import os
    import re
    import termios
    import tty

    __ANSICODE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

    __unconsumed_chars: List[str] = []

    # Adapted from: https://stackoverflow.com/a/59159112/9019559
    def __break_on_char(imput_str: str) -> List[str]:
        pos = 0
        result = []
        for m in __ANSICODE.finditer(imput_str):
            text = imput_str[pos : m.start()]
            result.extend(list(text))
            result.append(m.group())
            pos = m.end()

        text = imput_str[pos:]
        result.extend(list(text))
        return result

    def __get_key() -> str:
        fd_input = sys.stdin.fileno()
        term_attr = termios.tcgetattr(fd_input)
        fl = fcntl.fcntl(fd_input, fcntl.F_GETFL)
        fcntl.fcntl(fd_input, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        tty.setraw(fd_input)
        ch_str = ""
        try:
            while True:
                if ch_str != "":
                    break
                while True:
                    ch_stri = sys.stdin.read(1)
                    if ch_stri == "":
                        break
                    ch_str += ch_stri
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd_input, termios.TCSADRAIN, term_attr)
            fcntl.fcntl(fd_input, fcntl.F_SETFL, fl)
        first, *rest = __break_on_char(ch_str)
        __unconsumed_chars.extend(rest)
        return first

    def get_key() -> Key:
        """Returns a `Key` instance representing a keypress

        Returns:
            Key: Represents a keypress
        """
        if not __unconsumed_chars:
            ch_str = __get_key()
        else:
            ch_str = __unconsumed_chars.pop(0)

        ch_ord = tuple(map(ord, ch_str))
        return Key(ch_str, ch_ord, ch_str.isprintable() or ch_ord in [(13,), (27, 13)])


except ImportError:

    import msvcrt

    def get_key() -> Key:
        """Returns a `Key` instance representing a keypress

        Returns:
            Key: Represents a keypress
        """

        ch_str = msvcrt.getwch()
        if ch_str in ["à", "\x00"]:
            ch_str += msvcrt.getwch()
        ch_ord = tuple(map(ord, ch_str))
        return Key(ch_str, ch_ord, ch_str.isprintable())
