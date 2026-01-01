#!/usr/bin/env python3

# emby_wrapped_v2_1.py
# # Emby Wrapped (Spotify Wrapped-style year-end recap) for Emby user accounts
# # v2.1: robust type classification (no NaN.strip crash), Movies/TV Shows/TV Channels split,
# #       DHMS duration formatting, 12-hour peak hour, big HTML/CSS + chart presentation upgrade.

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import io
import json
import math
import os
import sqlite3
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import matplotlib.pyplot as plt

# # Optional: Rich console output (falls back cleanly)
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.traceback import install as rich_traceback_install

    rich_traceback_install(show_locals=True)
    _RICH = True
    console = Console()
except Exception:
    _RICH = False
    console = None


# ----------------------------
# # Config
# ----------------------------

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
    out_dir: str = "./emby_wrapped_v2_1_2025"
    max_top_titles: int = 10

    # # Assets
    include_posters: bool = True
    assets_dir: str = "./assets"

    # # Behavior
    http_timeout_seconds: int = 25
    poster_max_width: int = 900


def load_config(path: Path) -> Config:
    cfg = Config()
    if not path.exists():
        return cfg
    data = json.loads(path.read_text(encoding="utf-8"))
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ----------------------------
# # Utilities
# ----------------------------

def log(msg: str) -> None:
    if _RICH and console:
        console.print(msg)
    else:
        print(msg)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def rgb_tuple_to_css(rgb: Tuple[int, int, int]) -> str:
    return f"{rgb[0]},{rgb[1]},{rgb[2]}"


def format_duration_dhms(seconds: float) -> str:
    # # Always show days/hours/minutes/seconds
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
    # # 0 -> 12 AM, 12 -> 12 PM, 13 -> 1 PM
    h = int(hour) % 24
    ampm = "AM" if h < 12 else "PM"
    h12 = 12 if (h % 12) == 0 else (h % 12)
    return f"{h12} {ampm}"


DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def pick_palette(seed: str) -> Dict[str, Any]:
    # # Deterministic palette selection based on seed
    palettes = [
        {"name": "Sunset",  "accent": (255, 140, 66),  "accent2": (255, 72, 164), "bg1": (14, 10, 10), "bg2": (42, 16, 28)},
        {"name": "Neon",    "accent": (0, 224, 255),   "accent2": (255, 77, 240), "bg1": (8, 10, 18),  "bg2": (20, 10, 35)},
        {"name": "Aurora",  "accent": (78, 255, 168),  "accent2": (88, 171, 255), "bg1": (8, 14, 12), "bg2": (8, 20, 32)},
        {"name": "Citrus",  "accent": (255, 208, 66),  "accent2": (80, 255, 140), "bg1": (12, 12, 8), "bg2": (18, 24, 10)},
        {"name": "Grape",   "accent": (180, 120, 255), "accent2": (255, 95, 160), "bg1": (10, 8, 16), "bg2": (24, 10, 20)},
    ]
    h = 0
    for ch in seed:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return palettes[h % len(palettes)]


def mpl_apply_theme(theme: Dict[str, Any]) -> None:
    # # Dark-safe chart defaults
    bg = tuple(c / 255.0 for c in theme["bg1"])
    plt.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "text.color": (1, 1, 1, 0.92),
        "axes.labelcolor": (1, 1, 1, 0.92),
        "axes.edgecolor": (1, 1, 1, 0.25),
        "xtick.color": (1, 1, 1, 0.92),
        "ytick.color": (1, 1, 1, 0.92),
        "grid.color": (1, 1, 1, 0.14),
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
    })


def safe_isna(v: Any) -> bool:
    try:
        return pd.isna(v)
    except Exception:
        return False


def classify_media_type(emby_type: Any, itemtype_fallback: Any = None) -> str:
    # # Robust to pandas NaN / float values
    def _norm(x: Any) -> str:
        if x is None:
            return ""
        if safe_isna(x):
            return ""
        return str(x).strip().lower()

    t = _norm(emby_type) or _norm(itemtype_fallback)
    if not t:
        return "other"
    if t == "movie":
        return "movie"
    if t == "tvchannel":
        return "tv_channel"
    if t in {"episode", "series", "season"}:
        return "tv_show"
    return "other"


# ----------------------------
# # Emby API client
# ----------------------------

class EmbyClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base = cfg.server_url.rstrip("/") + cfg.base_path.rstrip("/")
        self.session = requests.Session()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.base + path
        params = dict(params or {})
        if self.cfg.api_key:
            params["api_key"] = self.cfg.api_key
        r = self.session.get(url, params=params, timeout=self.cfg.http_timeout_seconds)
        r.raise_for_status()
        return r.json()

    def fetch_users(self) -> pd.DataFrame:
        data = self._get("/Users")
        rows = []
        for u in data:
            rows.append({"UserId": u.get("Id"), "UserName": u.get("Name")})
        return pd.DataFrame(rows)

    def fetch_items_meta(self, item_ids: List[str]) -> pd.DataFrame:
        # # Chunked fetch. Emby supports Ids comma list.
        # # We request extra fields we care about for recap presentation.
        out: List[pd.DataFrame] = []
        chunk_size = 200
        for i in range(0, len(item_ids), chunk_size):
            chunk = item_ids[i:i + chunk_size]
            data = self._get(
                "/Items",
                params={
                    "Ids": ",".join(map(str, chunk)),
                    "Fields": ",".join([
                        "Genres",
                        "PrimaryImageTag",
                        "SeriesName",
                        "SeriesId",
                        "Type",
                        "Name",
                    ]),
                    "EnableImages": "true",
                },
            )
            items = data.get("Items") or []
            rows = []
            for it in items:
                rows.append({
                    "ItemId": str(it.get("Id")),
                    "Name": it.get("Name"),
                    "Type": it.get("Type"),
                    "SeriesName": it.get("SeriesName"),
                    "SeriesId": it.get("SeriesId"),
                    "Genres": it.get("Genres") or [],
                    "PrimaryImageTag": it.get("ImageTags", {}).get("Primary") or it.get("PrimaryImageTag"),
                })
            if rows:
                out.append(pd.DataFrame(rows))
        if not out:
            return pd.DataFrame(columns=["ItemId", "Name", "Type", "SeriesName", "SeriesId", "Genres", "PrimaryImageTag"])
        meta = pd.concat(out, ignore_index=True)
        meta["ItemId"] = meta["ItemId"].astype(str)
        return meta

    def download_primary_image(self, item_id: str, tag: Optional[str], max_width: int) -> Optional[bytes]:
        # # If tag is missing, Emby may still serve an image, but tag helps cache-busting.
        params = {"maxWidth": str(max_width), "quality": "90"}
        if self.cfg.api_key:
            params["api_key"] = self.cfg.api_key
        if tag:
            params["tag"] = tag

        url = self.base + f"/Items/{item_id}/Images/Primary"
        r = self.session.get(url, params=params, timeout=self.cfg.http_timeout_seconds)
        if r.status_code != 200 or not r.content:
            return None
        return r.content


# ----------------------------
# # Playback DB loading (auto-detect table/columns)
# ----------------------------

REQUIRED_KEYS = {
    "user": {"UserId"},
    "item": {"ItemId"},
    "date": {"DateCreated", "PlaybackStartTime", "Date", "StartTime"},
    "duration": {"PlayDuration", "PlaybackDuration", "Duration", "RunTimeSeconds"},
}

OPTIONAL_COLS = {
    "ItemType": {"ItemType", "Type"},
    "ItemName": {"ItemName", "Name"},
    "DeviceName": {"DeviceName", "Device"},
    "ClientName": {"ClientName", "Client"},
    "PlaybackMethod": {"PlaybackMethod", "Method"},
}


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return [r[1] for r in cur.fetchall()]


def choose_playback_table(conn: sqlite3.Connection) -> Tuple[str, Dict[str, str]]:
    # # Choose the best table by scoring matches to required/optional columns.
    best_table = ""
    best_score = -1
    best_map: Dict[str, str] = {}

    for t in list_tables(conn):
        cols = table_columns(conn, t)
        cols_set = set(cols)

        # # Find a matching column for each required category
        col_map: Dict[str, str] = {}
        ok = True
        score = 0

        # # required
        for key, candidates in REQUIRED_KEYS.items():
            found = None
            for c in candidates:
                if c in cols_set:
                    found = c
                    break
            if not found:
                ok = False
                break
            col_map[key] = found
            score += 5

        if not ok:
            continue

        # # optional
        for out_name, candidates in OPTIONAL_COLS.items():
            for c in candidates:
                if c in cols_set:
                    col_map[out_name] = c
                    score += 1
                    break

        # # prefer obvious playback tables
        name_l = t.lower()
        if "playback" in name_l:
            score += 2
        if "activity" in name_l:
            score += 1

        if score > best_score:
            best_score = score
            best_table = t
            best_map = col_map

    if not best_table:
        raise RuntimeError("Could not auto-detect playback table. Inspect tables/columns in your DB.")
    return best_table, best_map


def load_playback_df(cfg: Config) -> pd.DataFrame:
    db_path = Path(cfg.playback_db)
    if not db_path.exists():
        raise FileNotFoundError(f"Playback DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        table, cmap = choose_playback_table(conn)

        # # Build a SELECT with whatever columns exist
        select_cols = [
            f"{cmap['date']} AS DateCreated",
            f"{cmap['user']} AS UserId",
            f"{cmap['item']} AS ItemId",
            f"{cmap['duration']} AS PlayDuration",
        ]
        for out_name in ["ItemType", "ItemName", "DeviceName", "ClientName", "PlaybackMethod"]:
            if out_name in cmap:
                select_cols.append(f"{cmap[out_name]} AS {out_name}")
            else:
                select_cols.append(f"NULL AS {out_name}")

        # # Filter by year bounds (best-effort; if stored as text ISO, this works)
        start = dt.datetime(cfg.year, 1, 1, 0, 0, 0)
        end = dt.datetime(cfg.year + 1, 1, 1, 0, 0, 0)

        # # We do not assume storage is UTC vs local; we parse later with pandas.
        # # If the DB stores as an integer (ticks/epoch), SQLite string filter won't work.
        # # So we pull all and filter in pandas if needed.
        query = f"SELECT {', '.join(select_cols)} FROM {table}"
        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()

    # # Normalize types
    df["UserId"] = df["UserId"].astype(str)
    df["ItemId"] = df["ItemId"].astype(str)

    # # Parse DateCreated
    # # Handle ISO strings, naive strings, python datetime strings, and epoch-ish ints.
    # # Best effort: try datetime parse; if numeric, interpret as seconds (or ms) since epoch.
    dc = df["DateCreated"]

    if pd.api.types.is_numeric_dtype(dc):
        # # Guess seconds vs milliseconds by magnitude
        # # 10^12+ likely ms, 10^9 likely seconds
        s = pd.to_numeric(dc, errors="coerce")
        ms_mask = s > 1_000_000_000_000
        sec = s.copy()
        sec[ms_mask] = sec[ms_mask] / 1000.0
        ts = pd.to_datetime(sec, unit="s", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(dc, utc=True, errors="coerce")

        # # If parse failed massively, try without utc then localize
        if ts.isna().mean() > 0.80:
            ts2 = pd.to_datetime(dc, errors="coerce")
            ts = ts2.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")

    df["DateCreated"] = ts.dt.tz_convert(cfg.timezone)

    # # Duration: assume seconds
    df["PlayDuration"] = pd.to_numeric(df["PlayDuration"], errors="coerce").fillna(0).astype(float)
    df["Seconds"] = df["PlayDuration"]
    df["Minutes"] = df["Seconds"] / 60.0
    df["Hours"] = df["Minutes"] / 60.0

    # # Filter to year in local tz (after conversion)
    start_local = pd.Timestamp(dt.datetime(cfg.year, 1, 1), tz=cfg.timezone)
    end_local = pd.Timestamp(dt.datetime(cfg.year + 1, 1, 1), tz=cfg.timezone)
    df = df[(df["DateCreated"] >= start_local) & (df["DateCreated"] < end_local)].copy()

    # # Derived calendar fields
    df["Date"] = df["DateCreated"].dt.date.astype(str)
    df["DayOfWeek"] = df["DateCreated"].dt.day_name()
    df["Hour"] = df["DateCreated"].dt.hour.astype(int)
    df["Month"] = df["DateCreated"].dt.to_period("M").astype(str)

    # # Normalize optional cols
    for c in ["ItemType", "ItemName", "DeviceName", "ClientName", "PlaybackMethod"]:
        if c not in df.columns:
            df[c] = None

    return df


# ----------------------------
# # Metrics & aggregation
# ----------------------------

def build_display_title(merged: pd.DataFrame) -> pd.Series:
    # # Prefer SeriesName for TV shows, otherwise item name
    name_meta = merged.get("Name")
    if name_meta is None:
        name_meta = merged.get("ItemName")
    if name_meta is None:
        name_meta = pd.Series([""] * len(merged), index=merged.index)

    series = merged.get("SeriesName")
    if series is None:
        series = pd.Series([None] * len(merged), index=merged.index)

    display = pd.Series([""] * len(merged), index=merged.index, dtype="string")

    tv_mask = merged["Class"].eq("tv_show")
    display[tv_mask] = series.fillna("").astype("string")
    display[~tv_mask] = name_meta.fillna("").astype("string")

    # # If SeriesName is empty, fall back to name
    display = display.mask(display.str.len().eq(0), name_meta.fillna("").astype("string"))
    return display


def top_list(df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    s = (
        df.groupby("DisplayTitle")["Seconds"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    return [{"title": k, "seconds": float(v), "time": format_duration_dhms(float(v))} for k, v in s.items()]


def compute_metrics_for_user(df_u: pd.DataFrame, items_meta: pd.DataFrame, max_top_titles: int) -> Dict[str, Any]:
    # # Aggregate per item
    by_item = df_u.groupby("ItemId", as_index=False)["Seconds"].sum()

    merged = by_item.merge(items_meta, on="ItemId", how="left")

    # # Bring in fallback ItemType / ItemName from playback rows
    fallback = df_u[["ItemId", "ItemType", "ItemName"]].drop_duplicates(subset=["ItemId"])
    merged = merged.merge(fallback, on="ItemId", how="left", suffixes=("", "_fb"))

    # # Effective type: meta Type preferred, fallback to ItemType
    type_eff = merged["Type"].combine_first(merged["ItemType"])
    type_norm = type_eff.astype("string").fillna("").str.strip().str.lower()

    cls = pd.Series("other", index=merged.index)
    cls[type_norm.eq("movie")] = "movie"
    cls[type_norm.eq("tvchannel")] = "tv_channel"
    cls[type_norm.isin(["episode", "series", "season"])] = "tv_show"
    merged["Class"] = cls

    # # Effective name: meta Name preferred, fallback to ItemName
    merged["Name"] = merged["Name"].combine_first(merged["ItemName"])

    # # Display title (SeriesName for tv_show, otherwise name)
    merged["DisplayTitle"] = build_display_title(merged)

    # # Totals
    plays = int(len(df_u))
    distinct_titles = int(df_u["ItemId"].nunique())
    total_seconds = float(df_u["Seconds"].sum())

    # # Class totals
    class_sum = merged.groupby("Class")["Seconds"].sum().to_dict()
    movie_seconds = float(class_sum.get("movie", 0.0))
    tvshow_seconds = float(class_sum.get("tv_show", 0.0))
    tvch_seconds = float(class_sum.get("tv_channel", 0.0))
    other_seconds = float(class_sum.get("other", 0.0))

    # # Peak day
    dow_sum = df_u.groupby("DayOfWeek")["Seconds"].sum().to_dict()
    peak_dow = None
    if dow_sum:
        # # Choose best by seconds; stable ordering by DOW_ORDER
        peak_dow = sorted(dow_sum.items(), key=lambda kv: (kv[1], DOW_ORDER.index(kv[0]) if kv[0] in DOW_ORDER else 99))[-1][0]

    # # Peak hour
    hour_sum = df_u.groupby("Hour")["Seconds"].sum()
    peak_hour = int(hour_sum.idxmax()) if len(hour_sum) else 0

    # # Breakdown
    method_sum = (
        df_u.groupby("PlaybackMethod")["Seconds"].sum()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )
    client_sum = (
        df_u.groupby("ClientName")["Seconds"].sum()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    # # Series-level: (for tv shows) use DisplayTitle where Class == tv_show
    metrics = {
        "plays": plays,
        "distinct_titles": distinct_titles,
        "total_seconds": total_seconds,
        "total_time": format_duration_dhms(total_seconds),

        "movie_seconds": movie_seconds,
        "tvshow_seconds": tvshow_seconds,
        "tvch_seconds": tvch_seconds,
        "other_seconds": other_seconds,

        "movie_time": format_duration_dhms(movie_seconds),
        "tvshow_time": format_duration_dhms(tvshow_seconds),
        "tvch_time": format_duration_dhms(tvch_seconds),
        "other_time": format_duration_dhms(other_seconds),

        "peak_day": peak_dow or "—",
        "peak_hour": peak_hour,
        "peak_hour_label": format_hour_12h(peak_hour),

        "top_movies": top_list(merged[merged["Class"].eq("movie")], max_top_titles),
        "top_tvshows": top_list(merged[merged["Class"].eq("tv_show")], max_top_titles),
        "top_tvchannels": top_list(merged[merged["Class"].eq("tv_channel")], max_top_titles),

        "method_seconds": {str(k or "Unknown"): float(v) for k, v in method_sum.items()},
        "client_seconds": {str(k or "Unknown"): float(v) for k, v in client_sum.items()},
    }

    return metrics


def compute_distributions(df_u: pd.DataFrame) -> Dict[str, Any]:
    # # Hour distribution (0-23)
    hour = df_u.groupby("Hour")["Seconds"].sum().reindex(range(24), fill_value=0.0)

    # # DOW distribution in canonical order
    dow = df_u.groupby("DayOfWeek")["Seconds"].sum()
    dow = dow.reindex(DOW_ORDER, fill_value=0.0)

    # # Month distribution
    month = df_u.groupby("Month")["Seconds"].sum().sort_index()

    return {"hour": hour, "dow": dow, "month": month}


# ----------------------------
# # Chart rendering
# ----------------------------

def _save_fig_to_path(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_hour_chart(hour_s: pd.Series, theme: Dict[str, Any], out_path: Path) -> None:
    mpl_apply_theme(theme)
    accent = tuple(c / 255.0 for c in theme["accent"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.grid(True, axis="y")
    ax.bar(hour_s.index, hour_s.values / 3600.0, color=accent, alpha=0.90)
    ax.set_title("Watch Time by Hour")
    ax.set_ylabel("Hours")
    ax.set_xlabel("Hour")
    labels = [format_hour_12h(h) for h in range(24)]
    ax.set_xticks(list(range(24)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _save_fig_to_path(fig, out_path)


def render_dow_chart(dow_s: pd.Series, theme: Dict[str, Any], out_path: Path) -> None:
    mpl_apply_theme(theme)
    accent2 = tuple(c / 255.0 for c in theme["accent2"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.grid(True, axis="y")
    ax.bar(dow_s.index, dow_s.values / 3600.0, color=accent2, alpha=0.90)
    ax.set_title("Watch Time by Day of Week")
    ax.set_ylabel("Hours")
    ax.set_xlabel("Day")
    _save_fig_to_path(fig, out_path)


def render_month_chart(month_s: pd.Series, theme: Dict[str, Any], out_path: Path) -> None:
    mpl_apply_theme(theme)
    accent = tuple(c / 255.0 for c in theme["accent"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.grid(True, axis="y")
    ax.plot(month_s.index, month_s.values / 3600.0, marker="o", color=accent, linewidth=2.5, alpha=0.95)
    ax.set_title("Watch Time by Month")
    ax.set_ylabel("Hours")
    ax.set_xlabel("Month")
    ax.set_xticks(list(range(len(month_s.index))))
    ax.set_xticklabels(list(month_s.index), rotation=45, ha="right")
    _save_fig_to_path(fig, out_path)


# ----------------------------
# # HTML rendering
# ----------------------------

BASE_CSS = """
:root{
  --bg1: rgb(%(bg1)s);
  --bg2: rgb(%(bg2)s);
  --accent: rgb(%(accent)s);
  --accent2: rgb(%(accent2)s);

  --text: rgba(255,255,255,.92);
  --muted: rgba(255,255,255,.72);
  --card: rgba(255,255,255,.08);
  --card2: rgba(0,0,0,.22);
  --stroke: rgba(255,255,255,.14);
  --shadow: rgba(0,0,0,.35);
}

html,body{
  height:100%%;
  margin:0;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
  color: var(--text);
  background:
    radial-gradient(1200px 600px at 15%% 10%%, rgba(255,255,255,.08), transparent 60%%),
    radial-gradient(900px 500px at 85%% 25%%, rgba(255,255,255,.06), transparent 55%%),
    linear-gradient(145deg, var(--bg1), var(--bg2));
}

.container{
  max-width: 1120px;
  margin: 0 auto;
  padding: 28px 18px 60px;
}

.hero{
  padding: 22px 22px;
  border: 1px solid var(--stroke);
  background: linear-gradient(145deg, var(--card), rgba(255,255,255,.04));
  border-radius: 18px;
  box-shadow: 0 18px 40px var(--shadow);
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
}

.heroTop{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.hero h1{
  margin: 0;
  font-size: 34px;
  letter-spacing: -0.02em;
}

.sub{
  color: var(--muted);
  font-size: 14px;
  margin-top: 4px;
}

.badge{
  display:inline-block;
  padding: 5px 10px;
  border: 1px solid rgba(255,255,255,.16);
  border-radius: 999px;
  color: var(--muted);
  font-size: 12px;
}

.grid{
  display:grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 14px;
  margin-top: 14px;
}

.card{
  grid-column: span 4;
  border-radius: 18px;
  border: 1px solid var(--stroke);
  background: linear-gradient(145deg, var(--card), rgba(0,0,0,.12));
  padding: 16px 16px;
}

.card.wide{ grid-column: span 12; }
.card.half{ grid-column: span 6; }

.kpiRow{
  display:flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}

.kpiLabel{ color: var(--muted); font-size: 12px; }
.kpiValue{ font-size: 18px; font-weight: 750; }

.split3{
  display:grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.split3 .mini{
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 14px;
  padding: 12px 12px;
  background: linear-gradient(145deg, rgba(255,255,255,.06), rgba(0,0,0,.12));
}

.mini .t{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.mini .v{ font-size: 16px; font-weight: 750; }

.lists3{
  display:grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.list{
  margin: 8px 0 0;
  padding: 0;
  list-style: none;
}

.list li{
  display:flex;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255,255,255,.08);
}

.list li .title{
  overflow:hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 65%%;
}

.list li .time{
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}

img.chart{
  width: 100%%;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,.12);
  box-shadow: 0 12px 30px rgba(0,0,0,.30);
}

.poster{
  width: 100%%;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,.14);
  box-shadow: 0 16px 40px rgba(0,0,0,.35);
}
"""

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Emby Wrapped %(year)s — %(user)s</title>
  <style>%(css)s</style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <div class="heroTop">
        <div>
          <h1>Emby Wrapped <span style="color:var(--accent)">%(year)s</span></h1>
          <div class="sub">for <b>%(user)s</b> • peak hour: <b>%(peak_hour)s</b> • peak day: <b>%(peak_day)s</b></div>
        </div>
        <div class="badge">theme: %(theme_name)s</div>
      </div>

      %(poster_block)s
    </div>

    <div class="grid">
      <div class="card">
        <div class="kpiRow"><div class="kpiLabel">Total Playtime</div><div class="kpiValue">%(total_time)s</div></div>
        <div class="kpiRow"><div class="kpiLabel">Plays</div><div class="kpiValue">%(plays)s</div></div>
        <div class="kpiRow"><div class="kpiLabel">Distinct Titles</div><div class="kpiValue">%(distinct_titles)s</div></div>
      </div>

      <div class="card" style="grid-column: span 8;">
        <div class="kpiRow"><div class="kpiLabel">Playtime Split</div><div class="kpiValue">Movies / TV / Live</div></div>
        <div class="split3">
          <div class="mini">
            <div class="t">Movies</div>
            <div class="v">%(movie_time)s</div>
          </div>
          <div class="mini">
            <div class="t">TV Shows</div>
            <div class="v">%(tvshow_time)s</div>
          </div>
          <div class="mini">
            <div class="t">TV Channels</div>
            <div class="v">%(tvch_time)s</div>
          </div>
        </div>
        <div class="sub" style="margin-top:10px;">Other: <b>%(other_time)s</b></div>
      </div>

      <div class="card half">
        <div class="kpiRow"><div class="kpiLabel">Watch Time by Month</div><div class="kpiValue"></div></div>
        <img class="chart" src="%(month_chart)s" alt="month">
      </div>

      <div class="card half">
        <div class="kpiRow"><div class="kpiLabel">Watch Time by Hour</div><div class="kpiValue"></div></div>
        <img class="chart" src="%(hour_chart)s" alt="hour">
      </div>

      <div class="card wide">
        <div class="kpiRow"><div class="kpiLabel">Top Titles</div><div class="kpiValue"></div></div>
        <div class="lists3">
          <div>
            <div class="badge">Movies</div>
            <ul class="list">%(top_movies)s</ul>
          </div>
          <div>
            <div class="badge">TV Shows</div>
            <ul class="list">%(top_tvshows)s</ul>
          </div>
          <div>
            <div class="badge">TV Channels</div>
            <ul class="list">%(top_tvchannels)s</ul>
          </div>
        </div>
      </div>

      <div class="card wide">
        <div class="kpiRow"><div class="kpiLabel">Watch Time by Day of Week</div><div class="kpiValue"></div></div>
        <img class="chart" src="%(dow_chart)s" alt="dow">
      </div>
    </div>

    <div class="sub" style="margin-top:16px; text-align:center;">
      Generated on %(generated_on)s
    </div>
  </div>
</body>
</html>
"""


def li_rows(items: List[Dict[str, Any]]) -> str:
    if not items:
        return '<li><span class="title">—</span><span class="time">—</span></li>'
    out = []
    for it in items:
        title = (it.get("title") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        time_s = it.get("time") or "0d 00h 00m 00s"
        out.append(f'<li><span class="title">{title}</span><span class="time">{time_s}</span></li>')
    return "\n".join(out)


# ----------------------------
# # Build
# ----------------------------

def build(cfg: Config) -> None:
    out_root = Path(cfg.out_dir)
    ensure_dir(out_root)

    log(f"Reading playback DB: {cfg.playback_db}")
    df = load_playback_df(cfg)
    if df.empty:
        log(f"[yellow]No playback rows found for {cfg.year}.[/yellow]" if _RICH else f"No playback rows found for {cfg.year}.")
        return

    # # Emby client optional: if no api key, we can still build from DB-only data
    client = EmbyClient(cfg) if cfg.server_url and cfg.base_path else None

    # # User list
    user_map: Dict[str, str] = {}
    if client and cfg.api_key:
        log("Fetching user list...")
        try:
            users = client.fetch_users()
            user_map = {str(r["UserId"]): str(r["UserName"]) for _, r in users.iterrows()}
        except Exception as e:
            log(f"[yellow]Warning: could not fetch users from Emby: {e}[/yellow]" if _RICH else f"Warning: could not fetch users from Emby: {e}")

    # # Metadata enrichment
    items_meta = pd.DataFrame(columns=["ItemId", "Name", "Type", "SeriesName", "SeriesId", "Genres", "PrimaryImageTag"])
    all_item_ids = sorted(df["ItemId"].unique().tolist())

    if client and cfg.api_key:
        log("Enriching item metadata...")
        try:
            items_meta = client.fetch_items_meta(all_item_ids)
        except Exception as e:
            log(f"[yellow]Warning: could not fetch item metadata from Emby: {e}[/yellow]" if _RICH else f"Warning: could not fetch item metadata from Emby: {e}")

    # # Ensure ItemId is string in meta
    if not items_meta.empty:
        items_meta["ItemId"] = items_meta["ItemId"].astype(str)

    # # Per-user reports
    log("Building per-user reports...")

    user_ids = sorted(df["UserId"].unique().tolist())
    generated_on = dt.datetime.now().strftime("%Y-%m-%d %I:%M %p")

    if _RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("Users", total=len(user_ids))

            for uid in user_ids:
                prog.update(task, description=f"User {uid[:8]}…")
                _build_one_user(cfg, client, df, items_meta, uid, user_map, out_root, generated_on)
                prog.advance(task)
    else:
        for uid in user_ids:
            _build_one_user(cfg, client, df, items_meta, uid, user_map, out_root, generated_on)

    log(f"Done. Output: {out_root}")


def _build_one_user(
    cfg: Config,
    client: Optional[EmbyClient],
    df: pd.DataFrame,
    items_meta: pd.DataFrame,
    user_id: str,
    user_map: Dict[str, str],
    out_root: Path,
    generated_on: str,
) -> None:
    user_name = user_map.get(str(user_id), str(user_id))
    safe_name = "".join(ch for ch in user_name if ch.isalnum() or ch in ("-", "_")).strip() or str(user_id)

    user_out = out_root / safe_name
    ensure_dir(user_out)

    df_u = df[df["UserId"] == str(user_id)].copy()
    if df_u.empty:
        return

    theme = pick_palette(str(user_id))
    css = BASE_CSS % {
        "bg1": rgb_tuple_to_css(theme["bg1"]),
        "bg2": rgb_tuple_to_css(theme["bg2"]),
        "accent": rgb_tuple_to_css(theme["accent"]),
        "accent2": rgb_tuple_to_css(theme["accent2"]),
    }

    # # Metrics
    m = compute_metrics_for_user(df_u, items_meta, cfg.max_top_titles)

    # # Distributions & charts
    dist = compute_distributions(df_u)
    charts_dir = user_out / "charts"
    ensure_dir(charts_dir)

    month_chart = charts_dir / "month.png"
    hour_chart = charts_dir / "hour.png"
    dow_chart = charts_dir / "dow.png"

    render_month_chart(dist["month"], theme, month_chart)
    render_hour_chart(dist["hour"], theme, hour_chart)
    render_dow_chart(dist["dow"], theme, dow_chart)

    # # Poster (top item by seconds) optional
    poster_block = ""
    if cfg.include_posters and client and cfg.api_key and not items_meta.empty:
        try:
            # # Determine "hero" item by seconds (prefer movies, else anything)
            # # We'll pick the single most-watched ItemId in df_u.
            hero_item = (
                df_u.groupby("ItemId")["Seconds"].sum()
                .sort_values(ascending=False)
                .head(1)
            )
            if len(hero_item):
                hero_id = str(hero_item.index[0])
                row = items_meta[items_meta["ItemId"] == hero_id]
                tag = None
                title = None
                if not row.empty:
                    tag = row.iloc[0].get("PrimaryImageTag")
                    title = row.iloc[0].get("Name")
                img = client.download_primary_image(hero_id, tag, cfg.poster_max_width)
                if img:
                    b64 = base64.b64encode(img).decode("ascii")
                    poster_block = f'<img class="poster" src="data:image/jpeg;base64,{b64}" alt="poster">'
        except Exception:
            poster_block = ""

    # # Build HTML
    html = HTML_TEMPLATE % {
        "year": cfg.year,
        "user": user_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
        "css": css,
        "theme_name": theme["name"],

        "plays": m["plays"],
        "distinct_titles": m["distinct_titles"],
        "total_time": m["total_time"],

        "movie_time": m["movie_time"],
        "tvshow_time": m["tvshow_time"],
        "tvch_time": m["tvch_time"],
        "other_time": m["other_time"],

        "peak_day": m["peak_day"],
        "peak_hour": m["peak_hour_label"],

        "month_chart": str(month_chart.relative_to(user_out)).replace("\\", "/"),
        "hour_chart": str(hour_chart.relative_to(user_out)).replace("\\", "/"),
        "dow_chart": str(dow_chart.relative_to(user_out)).replace("\\", "/"),

        "top_movies": li_rows(m["top_movies"]),
        "top_tvshows": li_rows(m["top_tvshows"]),
        "top_tvchannels": li_rows(m["top_tvchannels"]),

        "poster_block": poster_block,
        "generated_on": generated_on,
    }

    (user_out / "index.html").write_text(html, encoding="utf-8")


# ----------------------------
# # CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Emby Wrapped year-end recap HTML per user from Playback Reporting DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python3 emby_wrapped_v2_1.py --config config.json
              python3 emby_wrapped_v2_1.py --playback-db ./playback_reporting.db --year 2025 --out ./out_2025

            Notes:
              - If api_key/server_url are provided, item metadata + posters are fetched from Emby.
              - If not, the recap still generates using DB-only columns (ItemType/ItemName).
            """
        ).strip(),
    )

    p.add_argument("--config", default="config.json", help="Path to config.json (default: ./config.json)")
    p.add_argument("--year", type=int, default=None, help="Override year")
    p.add_argument("--timezone", default=None, help="Override timezone (default: config value)")
    p.add_argument("--playback-db", default=None, help="Override playback_reporting.db path")
    p.add_argument("--out", default=None, help="Override output directory")
    p.add_argument("--max-top", type=int, default=None, help="Override max top titles per section")
    p.add_argument("--no-posters", action="store_true", help="Disable poster fetching/embedding")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    if args.year is not None:
        cfg.year = args.year
    if args.timezone is not None:
        cfg.timezone = args.timezone
    if args.playback_db is not None:
        cfg.playback_db = args.playback_db
    if args.out is not None:
        cfg.out_dir = args.out
    if args.max_top is not None:
        cfg.max_top_titles = args.max_top
    if args.no_posters:
        cfg.include_posters = False

    build(cfg)


if __name__ == "__main__":
    main()
