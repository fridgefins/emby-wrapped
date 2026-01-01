# # Config loader: JSON -> dataclass

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class Config:
    # # Emby
    server_url: str = "http://127.0.0.1:8096"
    base_path: str = "/emby"
    api_key: str = ""

    # # Data
    year: int = 2025
    timezone: str = "America/Chicago"
    playback_db: str = "./playback_reporting.db"

    # # Output
    out_dir: str = "./out"
    max_top_titles: int = 10

    # # Assets
    include_posters: bool = True
    poster_max_width: int = 900

    # # Network
    http_timeout_seconds: int = 25


def load_config(path: Path) -> Config:
    cfg = Config()
    if not path.exists():
        return cfg

    data = json.loads(path.read_text(encoding="utf-8"))
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
