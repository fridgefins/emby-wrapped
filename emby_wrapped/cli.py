# # CLI entrypoint: parse args, load config/layout, run orchestrator.

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .layout import load_layout
from .orchestrator import run


def main() -> None:
    p = argparse.ArgumentParser("emby-wrapped")

    # # Core IO
    p.add_argument("--config", default="config.json")
    p.add_argument("--layout", default="layout.json")
    p.add_argument("--playback-db", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--timezone", default=None)

    # # Feature toggles
    p.add_argument("--no-posters", action="store_true")

    args = p.parse_args()

    cfg = load_config(Path(args.config))
    layout = load_layout(Path(args.layout))

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

    run(cfg=cfg, layout=layout)
