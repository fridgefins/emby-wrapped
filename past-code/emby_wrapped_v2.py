#!/usr/bin/env python3
# emby_wrapped.py
#
# Emby Wrapped (v2)
# - Uses Playback Reporting plugin DB (playback_reporting.db)
# - Enriches metadata via Emby API
# - Produces per-user PNG "story pages" + optional PDF
#
# Key v2 upgrades:
# - Poster collage page (top 9 titles)
# - Signature stats "cards" page
# - Overview split: Movies vs TV Channels
# - Duration formatting in d/h/m/s
# - Peak hour in 12h time
# - Big presentation upgrade: themed palettes + dark-mode charts + inviting layouts

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import textwrap
from dataclasses import dataclass
from datetime import datetime, date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import typer
from dateutil import parser as dtparser
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from zoneinfo import ZoneInfo

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import portrait
except Exception:
    canvas = None


app = typer.Typer(add_completion=False)


# -------------------------
# Config
# -------------------------

@dataclass
class Config:
    server_url: str
    base_path: str
    api_key: str
    year: int
    timezone: str
    playback_db: str
    out_dir: str
    max_top_titles: int = 10
    include_posters: bool = True
    assets_dir: str = "./assets"  # Optional: place custom fonts in ./assets/fonts/*.ttf

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Config(
            server_url=data["server_url"].rstrip("/"),
            base_path=data.get("base_path", "/emby"),
            api_key=data["api_key"],
            year=int(data["year"]),
            timezone=data.get("timezone", "America/Chicago"),
            playback_db=data["playback_db"],
            out_dir=data["out_dir"],
            max_top_titles=int(data.get("max_top_titles", 10)),
            include_posters=bool(data.get("include_posters", True)),
            assets_dir=data.get("assets_dir", "./assets"),
        )


# -------------------------
# Theme / styling
# -------------------------

PALETTES = [
    # accent, accent2, bg1, bg2
    {"name": "Neon Pink", "accent": (255, 72, 164), "accent2": (0, 229, 255), "bg1": (10, 10, 16), "bg2": (26, 12, 40)},
    {"name": "Electric Lime", "accent": (155, 255, 72), "accent2": (255, 197, 0), "bg1": (10, 14, 10), "bg2": (14, 30, 18)},
    {"name": "Violet Sky", "accent": (167, 112, 255), "accent2": (66, 242, 217), "bg1": (10, 10, 16), "bg2": (12, 18, 42)},
    {"name": "Sunset", "accent": (255, 140, 66), "accent2": (255, 72, 164), "bg1": (14, 10, 10), "bg2": (42, 16, 28)},
    {"name": "Arctic", "accent": (72, 180, 255), "accent2": (167, 112, 255), "bg1": (8, 10, 16), "bg2": (10, 18, 34)},
]

def pick_palette(seed: str) -> Dict[str, Any]:
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(PALETTES)
    return PALETTES[idx]

def rgb(c: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    return (c[0], c[1], c[2], 255)

def hex_color(c: Tuple[int, int, int]) -> str:
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

def lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)

def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t))

def make_canvas(theme: Dict[str, Any], w: int = 1080, h: int = 1920) -> Image.Image:
    # Smooth vertical gradient + subtle blobs
    bg1 = theme["bg1"]
    bg2 = theme["bg2"]
    img = Image.new("RGBA", (w, h), rgb(bg1))
    d = ImageDraw.Draw(img)

    for y in range(h):
        t = y / max(h - 1, 1)
        c = lerp_color(bg1, bg2, t)
        d.line([(0, y), (w, y)], fill=rgb(c))

    # Add two large soft color blobs for depth
    blob = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(blob)

    a = theme["accent"]
    a2 = theme["accent2"]
    bd.ellipse([-250, 150, 650, 1050], fill=(a[0], a[1], a[2], 70))
    bd.ellipse([450, 800, 1350, 1700], fill=(a2[0], a2[1], a2[2], 70))
    blob = blob.filter(ImageFilter.GaussianBlur(80))
    img.alpha_composite(blob)
    return img

def rounded_rect(img: Image.Image, box: Tuple[int, int, int, int], radius: int, fill: Tuple[int, int, int, int], outline=None, width: int = 2) -> None:
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def draw_text(
    img: Image.Image,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int, int],
    shadow: bool = True,
    shadow_offset: Tuple[int, int] = (2, 2),
    shadow_alpha: int = 120,
    stroke: int = 0,
    stroke_fill: Tuple[int, int, int, int] = (0, 0, 0, 200),
) -> None:
    d = ImageDraw.Draw(img)
    x, y = xy
    if shadow:
        sx, sy = shadow_offset
        d.text((x + sx, y + sy), text, font=font, fill=(0, 0, 0, shadow_alpha))
    d.text((x, y), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)

def find_ttf_fonts(assets_dir: str) -> List[str]:
    fonts_dir = Path(assets_dir) / "fonts"
    if not fonts_dir.exists():
        return []
    return sorted(str(p) for p in fonts_dir.glob("*.ttf"))

def load_font(cfg: Config, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # Priority:
    # 1) assets/fonts/*.ttf (first is used for normal, second for bold if present)
    # 2) system fonts (DejaVu / Liberation / Noto)
    custom = find_ttf_fonts(cfg.assets_dir)
    if custom:
        try:
            if bold and len(custom) >= 2:
                return ImageFont.truetype(custom[1], size=size)
            return ImageFont.truetype(custom[0], size=size)
        except Exception:
            pass

    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


# -------------------------
# Time formatting
# -------------------------

def format_dhms(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    return f"{d}d {h:02d}h {m:02d}m {sec:02d}s"

def format_hour_12(hour_0_23: int) -> str:
    h = int(hour_0_23) % 24
    ampm = "AM" if h < 12 else "PM"
    hr = h % 12
    if hr == 0:
        hr = 12
    return f"{hr} {ampm}"

def shorten(s: str, n: int) -> str:
    return textwrap.shorten(str(s or ""), width=n, placeholder="…")


# -------------------------
# Emby client
# -------------------------

class EmbyClient:
    def __init__(self, server_url: str, base_path: str, api_key: str, timeout: int = 30):
        self.server_url = server_url.rstrip("/")
        self.base_path = (base_path or "").strip()
        if self.base_path and not self.base_path.startswith("/"):
            self.base_path = "/" + self.base_path
        self.api_key = api_key
        self.timeout = timeout
        self.sess = requests.Session()
        self.sess.headers.update({"X-Emby-Token": api_key})

    def _url(self, path: str) -> str:
        path = path if path.startswith("/") else "/" + path
        return f"{self.server_url}{self.base_path}{path}"

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = self.sess.get(self._url(path), params=params or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_bytes(self, path: str, params: Optional[Dict[str, Any]] = None) -> bytes:
        r = self.sess.get(self._url(path), params=params or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.content

    def get_users(self) -> pd.DataFrame:
        # Prefer /Users/Query; fallback /Users
        for endpoint in ("/Users/Query", "/Users"):
            try:
                data = self.get_json(endpoint)
                items = data.get("Items", data if isinstance(data, list) else [])
                if isinstance(items, list) and items:
                    rows = [{"UserId": u.get("Id"), "UserName": u.get("Name")} for u in items]
                    df = pd.DataFrame(rows).dropna()
                    if not df.empty:
                        return df
            except Exception:
                continue
        raise RuntimeError("Failed to retrieve users. Check API key permissions.")

    def get_items_by_ids(
        self,
        ids: List[str],
        fields: Optional[List[str]] = None,
        enable_images: bool = True,
    ) -> pd.DataFrame:
        params: Dict[str, Any] = {
            "Ids": ",".join(ids),
            "EnableImages": "true" if enable_images else "false",
            "EnableUserData": "false",
        }
        if fields:
            params["Fields"] = ",".join(fields)

        data = self.get_json("/Items", params=params)
        items = data.get("Items", [])
        if not items:
            return pd.DataFrame()

        rows = []
        for it in items:
            image_tags = it.get("ImageTags") or {}
            primary_tag = it.get("PrimaryImageTag") or image_tags.get("Primary")
            rows.append(
                {
                    "ItemId": it.get("Id"),
                    "Name": it.get("Name"),
                    "Type": it.get("Type"),
                    "SeriesName": it.get("SeriesName"),
                    "SeriesId": it.get("SeriesId"),
                    "Genres": it.get("Genres") or [],
                    "PrimaryImageTag": primary_tag,
                }
            )
        return pd.DataFrame(rows)

    def fetch_primary_image(
        self,
        item_id: str,
        tag: Optional[str],
        max_width: int = 700,
        max_height: int = 1008,
        quality: int = 90,
    ) -> Optional[Image.Image]:
        if not tag:
            return None
        params = {"MaxWidth": max_width, "MaxHeight": max_height, "Quality": quality, "Tag": tag}
        try:
            raw = self.get_bytes(f"/Items/{item_id}/Images/Primary", params=params)
            img = Image.open(BytesIO(raw)).convert("RGBA")
            return img
        except Exception:
            return None


# -------------------------
# Playback Reporting DB reader
# -------------------------

def sqlite_ro_connect(db_path: str) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)

def get_table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    cur = con.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def pick_first(existing: List[str], choices: List[str]) -> Optional[str]:
    s = set(existing)
    for c in choices:
        if c in s:
            return c
    return None

def read_playback_activity(db_path: str, year: int, tz: str) -> pd.DataFrame:
    start = f"{year:04d}-01-01"
    end = f"{year+1:04d}-01-01"

    con = sqlite_ro_connect(db_path)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            con,
        )["name"].tolist()

        if "PlaybackActivity" not in tables:
            raise RuntimeError(f"PlaybackActivity not found. Tables: {tables[:20]}...")

        cols = get_table_columns(con, "PlaybackActivity")

        c_date = pick_first(cols, ["DateCreated", "PlaybackStart", "StartTime"])
        c_user = pick_first(cols, ["UserId"])
        c_item = pick_first(cols, ["ItemId"])
        c_itemtype = pick_first(cols, ["ItemType"])
        c_itemname = pick_first(cols, ["ItemName", "Name"])
        c_duration = pick_first(cols, ["PlayDuration", "PlaybackDuration", "Duration", "PlayTime"])
        c_client = pick_first(cols, ["ClientName"])
        c_device = pick_first(cols, ["DeviceName"])
        c_method = pick_first(cols, ["PlaybackMethod"])

        required = [c_date, c_user, c_item]
        if any(x is None for x in required):
            raise RuntimeError(f"Missing required columns in PlaybackActivity. Found: {cols}")

        select_cols = [
            f"{c_date} AS DateCreated",
            f"{c_user} AS UserId",
            f"{c_item} AS ItemId",
        ]
        if c_itemtype: select_cols.append(f"{c_itemtype} AS ItemType")
        if c_itemname: select_cols.append(f"{c_itemname} AS ItemName")
        if c_duration: select_cols.append(f"{c_duration} AS PlayDuration")
        if c_client: select_cols.append(f"{c_client} AS ClientName")
        if c_device: select_cols.append(f"{c_device} AS DeviceName")
        if c_method: select_cols.append(f"{c_method} AS PlaybackMethod")

        q = f"""
        SELECT {", ".join(select_cols)}
        FROM PlaybackActivity
        WHERE DateCreated >= ? AND DateCreated < ?
        """
        df = pd.read_sql_query(q, con, params=[start, end])
    finally:
        con.close()

    if df.empty:
        return df

    z = ZoneInfo(tz)

    def parse_dt(x: Any) -> Optional[datetime]:
        if pd.isna(x):
            return None
        try:
            d = dtparser.parse(str(x))
            if d.tzinfo is None:
                d = d.replace(tzinfo=z)
            else:
                d = d.astimezone(z)
            return d
        except Exception:
            return None

    df["DateCreated"] = df["DateCreated"].map(parse_dt)
    df = df.dropna(subset=["DateCreated", "UserId", "ItemId"])

    # Normalize duration (seconds). Clamp invalid negatives.
    if "PlayDuration" in df.columns:
        df["PlayDuration"] = pd.to_numeric(df["PlayDuration"], errors="coerce").fillna(0)
        df["PlayDuration"] = df["PlayDuration"].clip(lower=0)
    else:
        df["PlayDuration"] = 0

    df["Seconds"] = df["PlayDuration"].astype(float)
    df["Minutes"] = df["Seconds"] / 60.0
    df["Hours"] = df["Seconds"] / 3600.0

    df["DayOfWeek"] = df["DateCreated"].dt.day_name()
    df["Hour"] = df["DateCreated"].dt.hour
    df["Date"] = df["DateCreated"].dt.date

    return df


# -------------------------
# Metrics
# -------------------------

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def classify_media_type(emby_type: Optional[str], itemtype_fallback: Optional[str]) -> str:
    t = (emby_type or itemtype_fallback or "").strip()
    if not t:
        return "other"
    # Movies
    if t.lower() == "movie":
        return "movie"
    # Live TV channels (what you asked for explicitly)
    if t.lower() in {"tvchannel", "channel", "livetvchannel"}:
        return "tv_channel"
    # Everything else
    return "other"

def compute_metrics_for_user(df_u: pd.DataFrame, items_meta: pd.DataFrame, max_top_titles: int) -> Dict[str, Any]:
    total_seconds = float(df_u["Seconds"].sum())
    plays = int(len(df_u))
    distinct_titles = int(df_u["ItemId"].nunique())

    by_item = (
        df_u.groupby(["ItemId", "ItemName"], dropna=False)["Seconds"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Enrich with Type for splits
    merged = df_u[["ItemId", "Seconds", "Hour", "DayOfWeek", "Date", "DeviceName", "ClientName", "PlaybackMethod"]].copy()
    merged = merged.merge(items_meta[["ItemId", "Type", "Genres", "SeriesId", "SeriesName"]], on="ItemId", how="left")

    # Type split: movie vs tv_channel vs other
    merged["Class"] = merged.apply(lambda r: classify_media_type(r.get("Type"), None), axis=1)
    class_sum = merged.groupby("Class")["Seconds"].sum().to_dict()
    movie_seconds = float(class_sum.get("movie", 0.0))
    tvch_seconds = float(class_sum.get("tv_channel", 0.0))
    other_seconds = float(total_seconds - movie_seconds - tvch_seconds)

    # Top title overall
    top_item_id = by_item.iloc[0]["ItemId"] if not by_item.empty else None
    top_item_name = by_item.iloc[0]["ItemName"] if not by_item.empty else None
    top_item_seconds = float(by_item.iloc[0]["Seconds"]) if not by_item.empty else 0.0

    # Series rollup (if available)
    series_roll = (
        merged.dropna(subset=["SeriesId"])
        .groupby(["SeriesId", "SeriesName"], dropna=False)["Seconds"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    top_series_name = series_roll.iloc[0]["SeriesName"] if not series_roll.empty else None
    top_series_seconds = float(series_roll.iloc[0]["Seconds"]) if not series_roll.empty else 0.0

    # Peak hour
    peak_hour = int(merged.groupby("Hour")["Seconds"].sum().sort_values(ascending=False).index[0]) if not merged.empty else 0

    # Heatmap DOW x 24 (minutes)
    heat = merged.groupby(["DayOfWeek", "Hour"])["Seconds"].sum().reset_index()
    heat["DayOfWeek"] = pd.Categorical(heat["DayOfWeek"], categories=DOW_ORDER, ordered=True)
    heat = heat.sort_values(["DayOfWeek", "Hour"])
    pivot = heat.pivot_table(index="DayOfWeek", columns="Hour", values="Seconds", fill_value=0).reindex(DOW_ORDER)
    pivot_minutes = pivot / 60.0

    # Genres (top 10)
    genres = []
    if "Genres" in merged.columns:
        g = merged.explode("Genres")
        g = g.dropna(subset=["Genres"])
        genres = (
            g.groupby("Genres")["Seconds"].sum().sort_values(ascending=False).head(10).reset_index()
        )
        genres["Minutes"] = genres["Seconds"] / 60.0
        genres = genres.to_dict("records")

    # Top movies / TV channels lists
    def top_by_class(cls: str) -> List[Dict[str, Any]]:
        sub = merged[merged["Class"] == cls].groupby("ItemId")["Seconds"].sum().sort_values(ascending=False).head(max_top_titles)
        if sub.empty:
            return []
        # Attach names
        names = df_u.drop_duplicates("ItemId").set_index("ItemId")["ItemName"].to_dict()
        out = []
        for item_id, sec in sub.items():
            out.append({"ItemId": item_id, "ItemName": names.get(item_id, item_id), "Seconds": float(sec)})
        return out

    top_movies = top_by_class("movie")
    top_tvch = top_by_class("tv_channel")

    # Devices / clients / methods
    def top_n(col: str, n: int = 6) -> List[Dict[str, Any]]:
        if col not in merged.columns:
            return []
        out = merged.groupby(col)["Seconds"].sum().sort_values(ascending=False).head(n).reset_index()
        out = out.rename(columns={col: "Name"})
        out["Seconds"] = out["Seconds"].astype(float)
        return out.to_dict("records")

    devices = top_n("DeviceName")
    clients = top_n("ClientName")
    methods = top_n("PlaybackMethod", n=10)

    # Signature stats
    nightowl_seconds = float(merged[merged["Hour"].isin([0, 1, 2, 3, 4])]["Seconds"].sum())
    weekend_seconds = float(merged[merged["DayOfWeek"].isin(["Saturday", "Sunday"])]["Seconds"].sum())
    binge_by_day = merged.groupby("Date")["Seconds"].sum().sort_values(ascending=False)
    binge_day = binge_by_day.index[0] if not binge_by_day.empty else None
    binge_day_seconds = float(binge_by_day.iloc[0]) if not binge_by_day.empty else 0.0
    longest_session_seconds = float(merged["Seconds"].max()) if not merged.empty else 0.0

    top_device = devices[0]["Name"] if devices else None

    # Direct vs transcode mix (by seconds)
    directish = 0.0
    transcode = 0.0
    if "PlaybackMethod" in merged.columns:
        for name, sec in merged.groupby("PlaybackMethod")["Seconds"].sum().items():
            n = str(name or "").lower()
            if "trans" in n:
                transcode += float(sec)
            elif "direct" in n:
                directish += float(sec)
            else:
                # Unknown: treat as direct-ish (often "DirectStream" or blank)
                directish += float(sec)

    return {
        "total_seconds": total_seconds,
        "plays": plays,
        "distinct_titles": distinct_titles,
        "movie_seconds": movie_seconds,
        "tvch_seconds": tvch_seconds,
        "other_seconds": other_seconds,
        "top_item_id": top_item_id,
        "top_item_name": top_item_name,
        "top_item_seconds": top_item_seconds,
        "top_series_name": top_series_name,
        "top_series_seconds": top_series_seconds,
        "peak_hour": peak_hour,
        "heatmap_minutes": pivot_minutes,
        "genres": genres,
        "devices": devices,
        "clients": clients,
        "methods": methods,
        "top_movies": top_movies,
        "top_tvch": top_tvch,
        "nightowl_seconds": nightowl_seconds,
        "weekend_seconds": weekend_seconds,
        "binge_day": binge_day,
        "binge_day_seconds": binge_day_seconds,
        "longest_session_seconds": longest_session_seconds,
        "top_device": top_device,
        "direct_seconds": directish,
        "transcode_seconds": transcode,
    }


# -------------------------
# Charts (dark-friendly)
# -------------------------

def fig_to_image(fig) -> Image.Image:
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    bio.seek(0)
    return Image.open(bio).convert("RGBA")

def mpl_dark_axes(ax, fg=(235, 235, 245), grid=(255, 255, 255, 35)):
    ax.tick_params(colors=hex_color(fg[:3]) if len(fg) == 4 else hex_color(fg))  # type: ignore
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.15))
    ax.title.set_color(hex_color(fg[:3]) if len(fg) == 4 else hex_color(fg))  # type: ignore
    ax.xaxis.label.set_color(hex_color(fg[:3]) if len(fg) == 4 else hex_color(fg))  # type: ignore
    ax.yaxis.label.set_color(hex_color(fg[:3]) if len(fg) == 4 else hex_color(fg))  # type: ignore
    ax.grid(True, axis="x", color=(grid[0]/255, grid[1]/255, grid[2]/255, grid[3]/255), linestyle="-", linewidth=1)
    ax.set_facecolor((0, 0, 0, 0))

def chart_barh_hours(labels: List[str], hours: List[float], title: str, theme: Dict[str, Any]) -> Image.Image:
    fig = plt.figure(figsize=(7.0, 4.6))
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0)

    accent = theme["accent"]
    c = hex_color(accent)

    labels_r = list(reversed(labels))
    hours_r = list(reversed(hours))
    ax.barh(labels_r, hours_r, color=c)

    ax.set_title(title)
    ax.set_xlabel("Hours")

    mpl_dark_axes(ax)
    fig.tight_layout()
    return fig_to_image(fig)

def chart_heatmap_minutes(pivot: pd.DataFrame, title: str, theme: Dict[str, Any]) -> Image.Image:
    fig = plt.figure(figsize=(9.0, 4.2))
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0)

    data = pivot.values
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([str(x) for x in range(0, 24, 2)])
    ax.tick_params(colors=hex_color((235, 235, 245)))
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.15))

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Minutes", color=hex_color((235, 235, 245)))
    cb.ax.yaxis.set_tick_params(color=hex_color((235, 235, 245)))
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=hex_color((235, 235, 245)))

    fig.tight_layout()
    return fig_to_image(fig)

def chart_pie_direct_vs_transcode(direct_s: float, trans_s: float, title: str, theme: Dict[str, Any]) -> Image.Image:
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    a = theme["accent"]
    a2 = theme["accent2"]
    vals = [max(0.0, direct_s), max(0.0, trans_s)]
    labels = ["Direct/Other", "Transcode"]
    colors = [hex_color(a), hex_color(a2)]

    if sum(vals) <= 0:
        vals = [1.0, 0.0]

    wedges, texts, autotexts = ax.pie(
        vals,
        labels=labels,
        autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
        startangle=90,
        colors=colors,
        textprops={"color": hex_color((235, 235, 245))},
    )
    for at in autotexts:
        at.set_color(hex_color((10, 10, 16)))

    ax.set_title(title, color=hex_color((235, 235, 245)))
    fig.tight_layout()
    return fig_to_image(fig)


# -------------------------
# Posters / collage helpers
# -------------------------

def cover_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    # Resize with center-crop to fill the tile
    src_w, src_h = img.size
    scale = max(w / src_w, h / src_h)
    nw, nh = int(src_w * scale), int(src_h * scale)
    im = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    return im.crop((left, top, left + w, top + h))

def poster_tile_placeholder(cfg: Config, theme: Dict[str, Any], title: str, w: int, h: int) -> Image.Image:
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bg = Image.new("RGBA", (w, h), (20, 20, 28, 220))
    tile.alpha_composite(bg)
    d = ImageDraw.Draw(tile)
    rounded_rect(tile, (0, 0, w, h), radius=24, fill=(20, 20, 28, 220), outline=(255, 255, 255, 25), width=2)

    font = load_font(cfg, 28, bold=True)
    txt = "\n".join(textwrap.wrap(shorten(title, 60), width=16))
    d.multiline_text((22, 22), txt, font=font, fill=(235, 235, 245, 255), spacing=6)
    return tile

def build_poster_collage(
    cfg: Config,
    client: EmbyClient,
    theme: Dict[str, Any],
    items_meta: pd.DataFrame,
    top_item_ids: List[str],
    tile_w: int = 300,
    tile_h: int = 450,
    cols: int = 3,
    pad: int = 18,
) -> Image.Image:
    rows = math.ceil(len(top_item_ids) / cols)
    w = cols * tile_w + (cols - 1) * pad
    h = rows * tile_h + (rows - 1) * pad

    collage = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    tag_map = {}
    if not items_meta.empty:
        sub = items_meta.set_index("ItemId")["PrimaryImageTag"].to_dict()
        tag_map.update({str(k): v for k, v in sub.items()})

    title_map = {}
    if not items_meta.empty:
        sub = items_meta.set_index("ItemId")["Name"].to_dict()
        title_map.update({str(k): v for k, v in sub.items()})

    for idx, item_id in enumerate(top_item_ids):
        r = idx // cols
        c = idx % cols
        x = c * (tile_w + pad)
        y = r * (tile_h + pad)

        tag = tag_map.get(str(item_id))
        poster = client.fetch_primary_image(str(item_id), tag, max_width=900, max_height=1400, quality=90) if cfg.include_posters else None
        if poster is None:
            poster = poster_tile_placeholder(cfg, theme, title_map.get(str(item_id), str(item_id)), tile_w, tile_h)
        else:
            poster = cover_resize(poster, tile_w, tile_h)

            # Round corners + subtle border
            mask = Image.new("L", (tile_w, tile_h), 0)
            md = ImageDraw.Draw(mask)
            md.rounded_rectangle((0, 0, tile_w, tile_h), radius=24, fill=255)
            rounded = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
            rounded.paste(poster, (0, 0), mask=mask)
            border = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
            bd = ImageDraw.Draw(border)
            bd.rounded_rectangle((1, 1, tile_w - 1, tile_h - 1), radius=24, outline=(255, 255, 255, 40), width=2)
            rounded.alpha_composite(border)
            poster = rounded

        collage.alpha_composite(poster, (x, y))

    return collage


# -------------------------
# Page renderers
# -------------------------

def paste_center(base: Image.Image, overlay: Image.Image, box: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ow, oh = overlay.size
    scale = min(bw / ow, bh / oh)
    nw, nh = int(ow * scale), int(oh * scale)
    o2 = overlay.resize((nw, nh), Image.LANCZOS)
    px = x0 + (bw - nw) // 2
    py = y0 + (bh - nh) // 2
    base.alpha_composite(o2, (px, py))

def render_overview(
    cfg: Config,
    theme: Dict[str, Any],
    user_name: str,
    year: int,
    metrics: Dict[str, Any],
    hero_poster: Optional[Image.Image],
) -> Image.Image:
    img = make_canvas(theme)
    a = theme["accent"]
    a2 = theme["accent2"]

    font_title = load_font(cfg, 68, bold=True)
    font_sub = load_font(cfg, 34, bold=False)
    font_big = load_font(cfg, 78, bold=True)
    font_mid = load_font(cfg, 40, bold=True)
    font_body = load_font(cfg, 32, bold=False)

    # Header
    draw_text(img, (70, 70), user_name, font_title, (245, 245, 255, 255))
    draw_text(img, (70, 150), f"Emby Wrapped • {year}", font_sub, (210, 210, 225, 255), shadow=False)

    # Big total time card
    card = (70, 220, 1010, 520)
    rounded_rect(img, card, radius=36, fill=(12, 12, 18, 170), outline=(255, 255, 255, 30), width=2)
    draw_text(img, (100, 250), "Total watch time", font_mid, (220, 220, 235, 255), shadow=False)

    total_str = format_dhms(metrics["total_seconds"])
    draw_text(img, (100, 320), total_str, font_big, (255, 255, 255, 255))

    # Split: Movies vs TV Channels
    left = (70, 560, 540, 780)
    right = (540, 560, 1010, 780)
    rounded_rect(img, left, radius=32, fill=(12, 12, 18, 150), outline=(255, 255, 255, 25), width=2)
    rounded_rect(img, right, radius=32, fill=(12, 12, 18, 150), outline=(255, 255, 255, 25), width=2)

    draw_text(img, (105, 590), "Movies", font_mid, rgb(a), shadow=False)
    draw_text(img, (105, 650), format_dhms(metrics["movie_seconds"]), font_body, (235, 235, 245, 255), shadow=False)

    draw_text(img, (575, 590), "TV Channels", font_mid, rgb(a2), shadow=False)
    draw_text(img, (575, 650), format_dhms(metrics["tvch_seconds"]), font_body, (235, 235, 245, 255), shadow=False)

    if metrics["other_seconds"] > 0:
        draw_text(img, (70, 800), f"Other: {format_dhms(metrics['other_seconds'])}", font_body, (210, 210, 225, 255), shadow=False)

    # Quick stats strip
    strip = (70, 870, 1010, 1040)
    rounded_rect(img, strip, radius=32, fill=(12, 12, 18, 130), outline=(255, 255, 255, 18), width=2)

    peak = format_hour_12(metrics["peak_hour"])
    draw_text(img, (100, 905), f"{metrics['plays']:,}", font_mid, (255, 255, 255, 255), shadow=False)
    draw_text(img, (100, 955), "plays", font_body, (210, 210, 225, 255), shadow=False)

    draw_text(img, (420, 905), f"{metrics['distinct_titles']:,}", font_mid, (255, 255, 255, 255), shadow=False)
    draw_text(img, (420, 955), "unique titles", font_body, (210, 210, 225, 255), shadow=False)

    draw_text(img, (720, 905), peak, font_mid, (255, 255, 255, 255), shadow=False)
    draw_text(img, (720, 955), "peak hour", font_body, (210, 210, 225, 255), shadow=False)

    # Top title / series
    draw_text(img, (70, 1100), "Top title", font_mid, (235, 235, 245, 255), shadow=False)
    draw_text(img, (70, 1160), shorten(metrics.get("top_item_name") or "—", 42), font_body, (255, 255, 255, 255), shadow=False)

    if metrics.get("top_series_name"):
        draw_text(img, (70, 1240), "Top series", font_mid, (235, 235, 245, 255), shadow=False)
        draw_text(img, (70, 1300), shorten(str(metrics["top_series_name"]), 42), font_body, (255, 255, 255, 255), shadow=False)

    # Hero poster
    if hero_poster is not None:
        hero = cover_resize(hero_poster, 360, 540)
        mask = Image.new("L", hero.size, 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, hero.size[0], hero.size[1]), radius=34, fill=255)
        rounded = Image.new("RGBA", hero.size, (0, 0, 0, 0))
        rounded.paste(hero, (0, 0), mask=mask)

        shadow = Image.new("RGBA", hero.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.rounded_rectangle((10, 10, hero.size[0], hero.size[1]), radius=34, fill=(0, 0, 0, 160))
        shadow = shadow.filter(ImageFilter.GaussianBlur(12))

        img.alpha_composite(shadow, (650, 1120))
        img.alpha_composite(rounded, (630, 1100))

    return img

def render_signature_cards(cfg: Config, theme: Dict[str, Any], user_name: str, year: int, m: Dict[str, Any]) -> Image.Image:
    img = make_canvas(theme)
    font_title = load_font(cfg, 62, bold=True)
    font_card_big = load_font(cfg, 44, bold=True)
    font_card_label = load_font(cfg, 28, bold=False)
    font_small = load_font(cfg, 24, bold=False)

    draw_text(img, (70, 70), "Your signature stats", font_title, (245, 245, 255, 255))
    draw_text(img, (70, 150), f"{user_name} • {year}", load_font(cfg, 30), (210, 210, 225, 255), shadow=False)

    # Cards layout: 2 columns x 3 rows
    cards = []
    x0, y0 = 70, 240
    card_w, card_h = 450, 260
    gap_x, gap_y = 40, 40

    def add_card(col: int, row: int, title: str, value: str, subtitle: str, accent: Tuple[int, int, int]):
        x = x0 + col * (card_w + gap_x)
        y = y0 + row * (card_h + gap_y)
        cards.append((x, y, title, value, subtitle, accent))

    # Compute useful strings
    night = format_dhms(m["nightowl_seconds"])
    weekend = format_dhms(m["weekend_seconds"])
    binge_day = str(m["binge_day"]) if m["binge_day"] else "—"
    binge_time = format_dhms(m["binge_day_seconds"])
    longest = format_dhms(m["longest_session_seconds"])
    top_device = shorten(m.get("top_device") or "—", 22)

    total = max(1.0, m["direct_seconds"] + m["transcode_seconds"])
    direct_pct = 100.0 * (m["direct_seconds"] / total)
    trans_pct = 100.0 * (m["transcode_seconds"] / total)

    add_card(0, 0, "Night owl", night, "watched between 12–5 AM", theme["accent"])
    add_card(1, 0, "Weekend", weekend, "time on Sat + Sun", theme["accent2"])
    add_card(0, 1, "Binge day", binge_day, f"most watched: {binge_time}", theme["accent2"])
    add_card(1, 1, "Longest session", longest, "single playback event", theme["accent"])
    add_card(0, 2, "Top device", top_device, "where you watched most", theme["accent"])
    add_card(1, 2, "Playback mix", f"{direct_pct:.0f}% / {trans_pct:.0f}%", "direct/other vs transcode", theme["accent2"])

    for (x, y, title, value, subtitle, accent) in cards:
        # Card background
        rounded_rect(img, (x, y, x + card_w, y + card_h), radius=36, fill=(12, 12, 18, 160), outline=(255, 255, 255, 25), width=2)

        # Accent stripe
        stripe = Image.new("RGBA", (card_w, 12), (accent[0], accent[1], accent[2], 255))
        img.alpha_composite(stripe, (x, y))

        draw_text(img, (x + 26, y + 24), title, load_font(cfg, 30, bold=True), (235, 235, 245, 255), shadow=False)
        draw_text(img, (x + 26, y + 80), value, font_card_big, (255, 255, 255, 255), shadow=False)
        draw_text(img, (x + 26, y + 170), subtitle, font_card_label, (210, 210, 225, 255), shadow=False)
        draw_text(img, (x + 26, y + 210), "—", font_small, (140, 140, 155, 255), shadow=False)

    return img

def render_poster_collage_page(
    cfg: Config,
    client: EmbyClient,
    theme: Dict[str, Any],
    items_meta: pd.DataFrame,
    user_name: str,
    year: int,
    top_item_ids: List[str],
) -> Image.Image:
    img = make_canvas(theme)
    draw_text(img, (70, 70), "Your year in posters", load_font(cfg, 62, bold=True), (245, 245, 255, 255))
    draw_text(img, (70, 150), f"{user_name} • {year}", load_font(cfg, 30), (210, 210, 225, 255), shadow=False)

    collage = build_poster_collage(cfg, client, theme, items_meta, top_item_ids[:9], tile_w=300, tile_h=450, cols=3, pad=18)

    # Frame around collage
    frame_box = (70, 240, 1010, 1780)
    rounded_rect(img, frame_box, radius=48, fill=(12, 12, 18, 120), outline=(255, 255, 255, 18), width=2)
    paste_center(img, collage, (90, 270, 990, 1750))

    return img

def render_top_movies_tv_page(cfg: Config, theme: Dict[str, Any], user_name: str, year: int, m: Dict[str, Any]) -> Image.Image:
    img = make_canvas(theme)
    draw_text(img, (70, 70), "Top picks", load_font(cfg, 62, bold=True), (245, 245, 255, 255))
    draw_text(img, (70, 150), f"{user_name} • {year}", load_font(cfg, 30), (210, 210, 225, 255), shadow=False)

    # Movies chart
    movies = m["top_movies"][:10]
    tvch = m["top_tvch"][:10]

    if movies:
        labels = [shorten(x["ItemName"], 42) for x in movies]
        hours = [x["Seconds"] / 3600.0 for x in movies]
        chart1 = chart_barh_hours(labels, hours, "Top movies (hours)", theme)
        rounded_rect(img, (70, 240, 1010, 980), radius=48, fill=(12, 12, 18, 120), outline=(255, 255, 255, 18), width=2)
        paste_center(img, chart1, (95, 270, 985, 960))
    else:
        rounded_rect(img, (70, 240, 1010, 520), radius=48, fill=(12, 12, 18, 120), outline=(255, 255, 255, 18), width=2)
        draw_text(img, (95, 300), "No movie plays recorded this year.", load_font(cfg, 34, bold=True), (235, 235, 245, 255), shadow=False)

    # TV Channels chart
    if tvch:
        labels = [shorten(x["ItemName"], 42) for x in tvch]
        hours = [x["Seconds"] / 3600.0 for x in tvch]
        chart2 = chart_barh_hours(labels, hours, "Top TV channels (hours)", theme)
        rounded_rect(img, (70, 1020, 1010, 1780), radius=48, fill=(12, 12, 18, 120), outline=(255, 255, 255, 18), width=2)
        paste_center(img, chart2, (95, 1050, 985, 1760))
    else:
        rounded_rect(img, (70, 1020, 1010, 1300), radius=48, fill=(12, 12, 18, 120), outline=(255, 255, 255, 18), width=2)
        draw_text(img, (95, 1090), "No TV channel plays recorded this year.", load_font(cfg, 34, bold=True), (235, 235, 245, 255), shadow=False)

    return img

def render_heatmap_page(cfg: Config, theme: Dict[str, Any], user_name: str, year: int, m: Dict[str, Any]) -> Image.Image:
    img = make_canvas(theme)
    draw_text(img, (70, 70), "When you watched", load_font(cfg, 62, bold=True), (245, 245, 255, 255))
    draw_text(img, (70, 150), f"{user_name} • {year}", load_font(cfg, 30), (210, 210, 225, 255), shadow=False)

    heat = chart_heatmap_minutes(m["heatmap_minutes"], "Minutes watched by day/hour", theme)
    rounded_rect(img, (70, 240, 1010, 1780), radius=48, fill=(12, 12, 18, 120), outline=(255, 255, 255, 18), width=2)
    paste_center(img, heat, (95, 270, 985, 1760))
    return img

def render_genres_page(cfg: Config, theme: Dict[str, Any], user_name: str, year: int, m: Dict[str, Any]) -> Optional[Image.Image]:
    if not m["genres"]:
        return None

    img = make_canvas(theme)
    draw_text(img, (70, 70), "Your top genres", load_font(cfg, 62, bold=True), (245, 245, 255, 255))
    draw_text(img, (70, 150), f"{user_name} • {year}", load_font(cfg, 30), (210, 210, 225, 255), shadow=False)

    labels = [shorten(x["Genres"], 26) for x in m["genres"][:10]]
    hours = [float(x["Seconds"]) / 3600.0 for x in m["genres"][:10]]
    chart = chart_barh_hours(labels, hours, "Genres by hours watched", theme)

    rounded_rect(img, (70, 240, 1010, 1780), radius=48, fill=(12, 12, 18, 120), outline=(255, 255, 255, 18), width=2)
    paste_center(img, chart, (95, 270, 985, 1760))
    return img

def render_devices_page(cfg: Config, theme: Dict[str, Any], user_name: str, year: int, m: Dict[str, Any]) -> Image.Image:
    img = make_canvas(theme)
    draw_text(img, (70, 70), "Devices & playback", load_font(cfg, 62, bold=True), (245, 245, 255, 255))
    draw_text(img, (70, 150), f"{user_name} • {year}", load_font(cfg, 30), (210, 210, 225, 255), shadow=False)

    # Left: top devices bar chart
    dev = m["devices"][:8]
    if dev:
        labels = [shorten(x["Name"], 22) for x in dev]
        hours = [float(x["Seconds"]) / 3600.0 for x in dev]
        dev_chart = chart_barh_hours(labels, hours, "Top devices (hours)", theme)
    else:
        dev_chart = None

    # Right: direct vs transcode pie
    pie = chart_pie_direct_vs_transcode(m["direct_seconds"], m["transcode_seconds"], "Direct vs Transcode", theme)

    # Layout cards
    rounded_rect(img, (70, 240, 700, 1780), radius=48, fill=(12, 12, 18, 120), outline=(255, 255, 255, 18), width=2)
    rounded_rect(img, (730, 240, 1010, 1780), radius=48, fill=(12, 12, 18, 120), outline=(255, 255, 255, 18), width=2)

    if dev_chart is not None:
        paste_center(img, dev_chart, (95, 270, 680, 1760))
    else:
        draw_text(img, (110, 320), "No device data.", load_font(cfg, 34, bold=True), (235, 235, 245, 255), shadow=False)

    paste_center(img, pie, (750, 310, 990, 820))

    # Also list top clients/methods as text below pie
    font = load_font(cfg, 26, bold=False)
    y = 870
    draw_text(img, (760, y), "Top clients", load_font(cfg, 30, bold=True), (235, 235, 245, 255), shadow=False)
    y += 50
    for row in m["clients"][:5]:
        draw_text(img, (760, y), f"• {shorten(row['Name'], 18)}", font, (210, 210, 225, 255), shadow=False)
        y += 36

    y += 30
    draw_text(img, (760, y), "Playback methods", load_font(cfg, 30, bold=True), (235, 235, 245, 255), shadow=False)
    y += 50
    for row in m["methods"][:6]:
        draw_text(img, (760, y), f"• {shorten(row['Name'], 18)}", font, (210, 210, 225, 255), shadow=False)
        y += 36

    return img


# -------------------------
# PDF
# -------------------------

def write_pdf(pdf_path: Path, pages: List[Path]) -> None:
    if canvas is None:
        return
    c = canvas.Canvas(str(pdf_path), pagesize=portrait((1080, 1920)))
    for p in pages:
        c.drawImage(str(p), 0, 0, width=1080, height=1920)
        c.showPage()
    c.save()


# -------------------------
# Main
# -------------------------

@app.command()
def build(config_path: str = typer.Option("config.json", help="Path to config.json")):
    cfg = Config.load(config_path)
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    client = EmbyClient(cfg.server_url, cfg.base_path, cfg.api_key)

    typer.echo(f"Reading playback DB: {cfg.playback_db}")
    df = read_playback_activity(cfg.playback_db, cfg.year, cfg.timezone)
    if df.empty:
        typer.echo("No playback rows found for that year.")
        raise typer.Exit(code=2)

    typer.echo("Fetching user list...")
    users = client.get_users()
    user_map = dict(zip(users["UserId"], users["UserName"]))

    # Enrich items: Type, Genres, SeriesId/Name, PrimaryImageTag
    typer.echo("Enriching item metadata...")
    item_ids = sorted(df["ItemId"].astype(str).unique().tolist())
    fields = ["Genres"]  # Keep light; Type/Name are base fields
    meta_chunks = []
    chunk_size = 200
    for i in range(0, len(item_ids), chunk_size):
        chunk = item_ids[i:i + chunk_size]
        try:
            meta = client.get_items_by_ids(chunk, fields=fields, enable_images=True)
            if not meta.empty:
                meta_chunks.append(meta)
        except Exception:
            continue
    items_meta = pd.concat(meta_chunks, ignore_index=True) if meta_chunks else pd.DataFrame(columns=["ItemId"])

    typer.echo("Building per-user reports...")
    for user_id, df_u in df.groupby("UserId"):
        user_name = user_map.get(user_id, f"unknown_{str(user_id)[:8]}")
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in user_name).strip("_") or "user"
        user_out = out_root / safe_name
        user_out.mkdir(parents=True, exist_ok=True)

        theme = pick_palette(str(user_id))

        # Metrics
        m = compute_metrics_for_user(df_u, items_meta, cfg.max_top_titles)

        # Hero poster (top item)
        hero = None
        if cfg.include_posters and m.get("top_item_id"):
            tag = None
            if not items_meta.empty:
                row = items_meta[items_meta["ItemId"] == str(m["top_item_id"])]
                if not row.empty:
                    tag = row.iloc[0].get("PrimaryImageTag")
            hero = client.fetch_primary_image(str(m["top_item_id"]), tag)

        # Poster collage IDs (top 9 overall by seconds)
        top9_ids = (
            df_u.groupby("ItemId")["Seconds"].sum().sort_values(ascending=False).head(9).index.astype(str).tolist()
        )

        # Render pages
        pages_img: List[Tuple[str, Image.Image]] = []
        pages_img.append(("page_01_overview.png", render_overview(cfg, theme, user_name, cfg.year, m, hero)))
        pages_img.append(("page_02_signature_stats.png", render_signature_cards(cfg, theme, user_name, cfg.year, m)))
        pages_img.append(("page_03_poster_collage.png", render_poster_collage_page(cfg, client, theme, items_meta, user_name, cfg.year, top9_ids)))
        pages_img.append(("page_04_top_movies_tvch.png", render_top_movies_tv_page(cfg, theme, user_name, cfg.year, m)))
        pages_img.append(("page_05_heatmap.png", render_heatmap_page(cfg, theme, user_name, cfg.year, m)))

        gpage = render_genres_page(cfg, theme, user_name, cfg.year, m)
        if gpage is not None:
            pages_img.append(("page_06_genres.png", gpage))

        pages_img.append(("page_07_devices.png", render_devices_page(cfg, theme, user_name, cfg.year, m)))

        out_paths: List[Path] = []
        for fname, im in pages_img:
            p = user_out / fname
            im.save(p)
            out_paths.append(p)

        # Save raw metrics for debugging / iteration
        with open(user_out / "report.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "user_id": str(user_id),
                    "user_name": user_name,
                    "year": cfg.year,
                    **{k: v for k, v in m.items() if not isinstance(v, pd.DataFrame)},
                },
                f,
                indent=2,
                default=str,
            )

        # PDF
        pdf_path = user_out / f"emby_wrapped_{cfg.year}.pdf"
        write_pdf(pdf_path, out_paths)

        typer.echo(f"✅ {user_name}: {len(out_paths)} pages -> {user_out}")

    typer.echo("Done.")


if __name__ == "__main__":
    app()
