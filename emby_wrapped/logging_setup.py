from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

from .console import make_console, ConsoleOptions


def setup_logging(
    *,
    level: str = "INFO",
    log_file: str | None = None,
    console_width: int | None = None,
    no_color: bool = False,
) -> None:
    """
    Configure root logging with RichHandler for neat, colorful console logs.
    Optionally also log to a file.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    console = make_console(ConsoleOptions(width=console_width, no_color=no_color))
    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
    ]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(fh)

    logging.basicConfig(level=numeric_level, handlers=handlers)
