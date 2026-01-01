# # CLI entrypoint: parse args, load config/layout, run orchestrator.

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data.playback_db import load_playback_df
from .layout import load_layout
from .orchestrator import run
from .logging_setup import setup_logging
from .data.playback_db import load_playback_df
from .debug_views import render_debug_sample

def main() -> None:
    p = argparse.ArgumentParser("emby-wrapped")

    # # Core IO
    p.add_argument("--config", default="config.json")
    p.add_argument("--layout", default="layout.json")
    p.add_argument("--playback-db", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--timezone", default=None)

    # # CLI
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--log-file", default=None)
    p.add_argument("--console-width", type=int, default=None)
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--debug-sample", action="store_true", help="Prints sample rows and exits")

    # # Feature toggles
    p.add_argument("--no-posters", action="store_true")

    args = p.parse_args()

    cfg = load_config(Path(args.config))
    layout = load_layout(Path(args.layout))

    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        console_width=args.console_width,
        no_color=args.no_color,
    )

    # # Overrides
    if args.playback_db is not None:
        cfg.playback_db = args.playback_db
    if args.out is not None:
        cfg.out_dir = args.out
    if args.year is not None:
        cfg.year = args.year
    if args.timezone is not None:
        cfg.timezone = args.timezone
    if args.no_posters:
        cfg.include_posters = False
    if args.debug_sample:
        df = load_playback_df(playback_db=cfg.playback_db, year=cfg.year, timezone=cfg.timezone)
        render_debug_sample(df, width=args.console_width, no_color=args.no_color)
        raise SystemExit(0)

    run(cfg=cfg, layout=layout)
