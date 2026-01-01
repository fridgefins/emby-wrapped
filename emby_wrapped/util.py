# # Shared helpers: duration formatting, safe string normalization, etc.

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def safe_isna(v: Any) -> bool:
    try:
        return pd.isna(v)
    except Exception:
        return False


def norm_str(v: Any) -> str:
    # # Robust against pandas NaN / float values
    if v is None:
        return ""
    if safe_isna(v):
        return ""
    return str(v).strip()


def format_duration_dhms(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    try:
        if isinstance(seconds, float) and math.isnan(seconds):
            seconds = 0.0
    except Exception:
        pass

    total = int(round(float(seconds)))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}d {hours:02}h {minutes:02}m {secs:02}s"


def format_hour_12h(hour: int) -> str:
    h = int(hour) % 24
    ampm = "AM" if h < 12 else "PM"
    h12 = 12 if (h % 12) == 0 else (h % 12)
    return f"{h12} {ampm}"


DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
