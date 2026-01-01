from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from rich.console import Console
from rich.theme import Theme as RichTheme


@dataclass(frozen=True)
class ConsoleOptions:
    # If None, use terminal size.
    width: int | None = None
    # If True, disable color.
    no_color: bool = False


def make_console(opts: ConsoleOptions | None = None) -> Console:
    opts = opts or ConsoleOptions()

    # Allow override via env var (useful for CI / piping to files)
    env_width = os.getenv("EMBY_WRAPPED_CONSOLE_WIDTH")
    width = opts.width
    if width is None and env_width:
        try:
            width = int(env_width)
        except ValueError:
            width = None

    if width is None:
        width = shutil.get_terminal_size(fallback=(140, 40)).columns

    theme = RichTheme(
        {
            "info": "cyan",
            "ok": "green",
            "warn": "yellow",
            "err": "red",
            "dim": "dim",
            "k": "bold",
        }
    )

    return Console(
        width=width,
        theme=theme,
        color_system=None if opts.no_color else "auto",
        # soft_wrap=False is important: we want truncation/ellipsis in tables,
        # not automatic wrapping by Console.
        soft_wrap=False,
        highlight=False,
    )
