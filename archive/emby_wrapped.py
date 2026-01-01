#!/usr/bin/env python3
# emby_wrapped.py
#
# Generates "Wrapped"-style infographics per Emby user by reading the Playback Reporting plugin DB
# and enriching item metadata via the Emby REST API.

from __future__ import annotations

import json
import math
import os
import sqlite3
import textwrap
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import typer
from dateutil import parser as dtparser
from PIL import Image, ImageDraw, ImageFont

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
        )


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
        # Emby has /Users/Query which returns a list of users (admin scope). :contentReference[oaicite:8]{index=8}
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
        raise RuntimeError("Failed to retrieve users via /Users/Query or /Users. Check API key permissions.")

    def get_items_by_ids(
        self,
        ids: List[str],
        user_id: Optional[str] = None,
        fields: Optional[List[str]] = None,
        enable_images: bool = True,
        enable_userdata: bool = False,
    ) -> pd.DataFrame:
        # /Items supports Ids=... and Fields=... :contentReference[oaicite:9]{index=9}
        params: Dict[str, Any] = {
            "Ids": ",".join(ids),
            "EnableImages": "true" if enable_images else "false",
            "EnableUserData": "true" if enable_userdata else "false",
        }
        if user_id:
            params["UserId"] = user_id
        if fields:
            params["Fields"] = ",".join(fields)

        data = self.get_json("/Items", params=params)
        items = data.get("Items", [])
        if not items:
            return pd.DataFrame()

        rows = []
        for it in items:
            rows.append(
                {
                    "ItemId": it.get("Id"),
                    "Name": it.get("Name"),
                    "Type": it.get("Type"),
                    "SeriesName": it.get("SeriesName"),
                    "SeriesId": it.get("SeriesId"),
                    "Genres": it.get("Genres") or [],
                    "PrimaryImageTag": it.get("PrimaryImageTag"),
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
        # ImageService supports /Items/{Id}/Images/{Type} (Type=Primary). :contentReference[oaicite:10]{index=10}
        # Emby docs recommend only downloading if tag exists (otherwise often 404). :contentReference[oaicite:11]{index=11}
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
    # Read-only, safer for automation
    # If you copied the DB elsewhere, this avoids server locks anyway.
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
        c_remote = pick_first(cols, ["RemoteAddress"])

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
        if c_remote: select_cols.append(f"{c_remote} AS RemoteAddress")

        # DateCreated is typically ISO-like text; range filters work well lexicographically
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

    # Normalize duration (seconds). Some installs store bad negatives; clamp. :contentReference[oaicite:12]{index=12}
    if "PlayDuration" in df.columns:
        df["PlayDuration"] = pd.to_numeric(df["PlayDuration"], errors="coerce").fillna(0)
        df["PlayDuration"] = df["PlayDuration"].clip(lower=0)
    else:
        df["PlayDuration"] = 0

    df["Minutes"] = df["PlayDuration"] / 60.0

    df["DayOfWeek"] = df["DateCreated"].dt.day_name()
    df["Hour"] = df["DateCreated"].dt.hour
    df["Date"] = df["DateCreated"].dt.date

    return df


# -------------------------
# Metrics
# -------------------------

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def compute_metrics_for_user(df_u: pd.DataFrame, items_meta: pd.DataFrame) -> Dict[str, Any]:
    total_minutes = float(df_u["Minutes"].sum())
    plays = int(len(df_u))
    distinct_titles = int(df_u["ItemId"].nunique())

    by_item = (
        df_u.groupby(["ItemId", "ItemName"], dropna=False)["Minutes"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    top_item_id = by_item.iloc[0]["ItemId"] if not by_item.empty else None
    top_item_name = by_item.iloc[0]["ItemName"] if not by_item.empty else None
    top_item_minutes = float(by_item.iloc[0]["Minutes"]) if not by_item.empty else 0.0

    # Series rollup if SeriesId is available in metadata
    merged = df_u[["ItemId", "Minutes"]].merge(items_meta, on="ItemId", how="left")
    series_roll = (
        merged.dropna(subset=["SeriesId"])
        .groupby(["SeriesId", "SeriesName"], dropna=False)["Minutes"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    top_series_name = series_roll.iloc[0]["SeriesName"] if not series_roll.empty else None
    top_series_minutes = float(series_roll.iloc[0]["Minutes"]) if not series_roll.empty else 0.0

    # Peak hour
    peak_hour = int(df_u.groupby("Hour")["Minutes"].sum().sort_values(ascending=False).index[0]) if not df_u.empty else 0

    # Heatmap matrix: DOW x 24
    heat = (
        df_u.groupby(["DayOfWeek", "Hour"])["Minutes"].sum().reset_index()
    )
    heat["DayOfWeek"] = pd.Categorical(heat["DayOfWeek"], categories=DOW_ORDER, ordered=True)
    heat = heat.sort_values(["DayOfWeek", "Hour"])
    pivot = heat.pivot_table(index="DayOfWeek", columns="Hour", values="Minutes", fill_value=0).reindex(DOW_ORDER)

    # Genres (requires metadata enrichment)
    genres = []
    if not items_meta.empty and "Genres" in items_meta.columns:
        g = merged.explode("Genres")
        g = g.dropna(subset=["Genres"])
        genres = (
            g.groupby("Genres")["Minutes"].sum().sort_values(ascending=False).head(10).reset_index().to_dict("records")
        )

    # Devices/clients/method
    def top_n(col: str, n: int = 5) -> List[Dict[str, Any]]:
        if col not in df_u.columns:
            return []
        out = (
            df_u.groupby(col)["Minutes"].sum().sort_values(ascending=False).head(n).reset_index()
        )
        out = out.rename(columns={col: "Name"})
        return out.to_dict("records")

    devices = top_n("DeviceName")
    clients = top_n("ClientName")
    methods = top_n("PlaybackMethod", n=10)

    # Daily streak
    days = sorted(set(df_u["Date"].tolist()))
    streak = 0
    best = 0
    for i in range(len(days)):
        if i == 0:
            streak = 1
        else:
            delta = (days[i] - days[i - 1]).days
            if delta == 1:
                streak += 1
            else:
                streak = 1
        best = max(best, streak)

    return {
        "total_minutes": total_minutes,
        "plays": plays,
        "distinct_titles": distinct_titles,
        "top_item_id": top_item_id,
        "top_item_name": top_item_name,
        "top_item_minutes": top_item_minutes,
        "top_series_name": top_series_name,
        "top_series_minutes": top_series_minutes,
        "peak_hour": peak_hour,
        "heatmap": pivot,
        "top_titles": by_item.head(10).to_dict("records"),
        "genres": genres,
        "devices": devices,
        "clients": clients,
        "methods": methods,
        "best_streak_days": best,
    }


# -------------------------
# Rendering helpers (PIL + matplotlib)
# -------------------------

def load_font(size: int) -> ImageFont.FreeTypeFont:
    # Debian typically has DejaVu installed; fallback to default if not found.
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def wrap_text(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s or "", width=width))

def fig_to_image(fig) -> Image.Image:
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    bio.seek(0)
    return Image.open(bio).convert("RGBA")

def chart_bar(labels: List[str], values: List[float], title: str) -> Image.Image:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.barh(list(reversed(labels)), list(reversed(values)))
    ax.set_title(title)
    ax.set_xlabel("Minutes")
    fig.tight_layout()
    return fig_to_image(fig)

def chart_heatmap(pivot: pd.DataFrame, title: str) -> Image.Image:
    fig = plt.figure(figsize=(8, 3.8))
    ax = fig.add_subplot(111)
    data = pivot.values
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([str(x) for x in range(0, 24, 2)])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Minutes")
    fig.tight_layout()
    return fig_to_image(fig)

def make_canvas(w: int = 1080, h: int = 1920) -> Image.Image:
    # Simple dark gradient background
    img = Image.new("RGBA", (w, h), (10, 10, 14, 255))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        v = int(10 + (y / h) * 35)
        draw.line([(0, y), (w, y)], fill=(v, v, v + 10, 255))
    return img

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

def render_pages(
    out_dir: Path,
    user_name: str,
    year: int,
    metrics: Dict[str, Any],
    top_poster: Optional[Image.Image],
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    font_big = load_font(90)
    font_h1 = load_font(56)
    font_h2 = load_font(40)
    font_body = load_font(32)

    pages: List[Path] = []

    # Page 1: Overview
    img = make_canvas()
    d = ImageDraw.Draw(img)

    d.text((70, 70), f"{user_name}", font=font_h1, fill=(255, 255, 255, 255))
    d.text((70, 135), f"Emby Wrapped • {year}", font=font_h2, fill=(200, 200, 210, 255))

    mins = metrics["total_minutes"]
    d.text((70, 240), f"{mins:,.0f}", font=font_big, fill=(255, 255, 255, 255))
    d.text((70, 350), "minutes watched", font=font_h2, fill=(210, 210, 220, 255))

    d.text((70, 450), f"{metrics['plays']:,} plays", font=font_h2, fill=(230, 230, 240, 255))
    d.text((70, 510), f"{metrics['distinct_titles']:,} unique titles", font=font_h2, fill=(230, 230, 240, 255))
    d.text((70, 570), f"best streak: {metrics['best_streak_days']} days", font=font_h2, fill=(230, 230, 240, 255))
    d.text((70, 630), f"peak hour: {metrics['peak_hour']:02d}:00", font=font_h2, fill=(230, 230, 240, 255))

    top_title = metrics.get("top_item_name") or "—"
    top_title = wrap_text(top_title, width=26)
    d.text((70, 730), "top title:", font=font_h2, fill=(200, 200, 210, 255))
    d.text((70, 780), top_title, font=font_body, fill=(255, 255, 255, 255))

    if metrics.get("top_series_name"):
        s = wrap_text(str(metrics["top_series_name"]), width=28)
        d.text((70, 900), "top series:", font=font_h2, fill=(200, 200, 210, 255))
        d.text((70, 950), s, font=font_body, fill=(255, 255, 255, 255))

    # Poster (optional)
    if top_poster is not None:
        paste_center(img, top_poster, (620, 650, 1020, 1250))

    p1 = out_dir / "page_01_overview.png"
    img.save(p1)
    pages.append(p1)

    # Page 2: Top titles
    titles = metrics["top_titles"]
    labels = [t.get("ItemName") or t.get("ItemId") for t in titles]
    values = [float(t.get("Minutes", 0)) for t in titles]
    labels = [textwrap.shorten(str(x), width=40, placeholder="…") for x in labels]

    chart = chart_bar(labels, values, title="Top titles by minutes")
    img = make_canvas()
    d = ImageDraw.Draw(img)
    d.text((70, 70), f"Top titles • {year}", font=font_h1, fill=(255, 255, 255, 255))
    paste_center(img, chart, (70, 180, 1010, 1700))
    p2 = out_dir / "page_02_top_titles.png"
    img.save(p2)
    pages.append(p2)

    # Page 3: Heatmap
    heat_img = chart_heatmap(metrics["heatmap"], title="When you watched (minutes)")
    img = make_canvas()
    d = ImageDraw.Draw(img)
    d.text((70, 70), f"Viewing heatmap • {year}", font=font_h1, fill=(255, 255, 255, 255))
    paste_center(img, heat_img, (70, 220, 1010, 1550))
    p3 = out_dir / "page_03_heatmap.png"
    img.save(p3)
    pages.append(p3)

    # Page 4: Genres
    if metrics["genres"]:
        g_labels = [g["Genres"] for g in metrics["genres"]]
        g_vals = [float(g["Minutes"]) for g in metrics["genres"]]
        g_chart = chart_bar(g_labels, g_vals, title="Top genres by minutes")
        img = make_canvas()
        d = ImageDraw.Draw(img)
        d.text((70, 70), f"Genres • {year}", font=font_h1, fill=(255, 255, 255, 255))
        paste_center(img, g_chart, (70, 180, 1010, 1700))
        p4 = out_dir / "page_04_genres.png"
        img.save(p4)
        pages.append(p4)

    # Page 5: Devices & playback method
    dev = metrics["devices"]
    meth = metrics["methods"]

    img = make_canvas()
    d = ImageDraw.Draw(img)
    d.text((70, 70), f"Devices • {year}", font=font_h1, fill=(255, 255, 255, 255))

    y = 200
    d.text((70, y), "Top devices:", font=font_h2, fill=(220, 220, 230, 255))
    y += 60
    for row in dev[:6]:
        name = row["Name"] or "—"
        mins = float(row["Minutes"])
        d.text((90, y), f"• {name} — {mins:,.0f} min", font=font_body, fill=(255, 255, 255, 255))
        y += 46

    y += 40
    d.text((70, y), "Playback methods:", font=font_h2, fill=(220, 220, 230, 255))
    y += 60
    for row in meth[:6]:
        name = row["Name"] or "—"
        mins = float(row["Minutes"])
        d.text((90, y), f"• {name} — {mins:,.0f} min", font=font_body, fill=(255, 255, 255, 255))
        y += 46

    p5 = out_dir / "page_05_devices.png"
    img.save(p5)
    pages.append(p5)

    return pages

def write_pdf(pdf_path: Path, pages: List[Path]) -> None:
    if canvas is None:
        return
    c = canvas.Canvas(str(pdf_path), pagesize=portrait((1080, 1920)))
    for p in pages:
        c.drawImage(str(p), 0, 0, width=1080, height=1920)
        c.showPage()
    c.save()


# -------------------------
# Main command
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
        raise typer.Exit(code=2)

    typer.echo("Fetching user list...")
    users = client.get_users()
    user_map = dict(zip(users["UserId"], users["UserName"]))

    # Enrich items (Genres, SeriesName/Id, PrimaryImageTag)
    typer.echo("Enriching item metadata...")
    item_ids = sorted(df["ItemId"].astype(str).unique().tolist())
    fields = ["Genres", "PrimaryImageAspectRatio", "ProviderIds"]  # Genres is the big one :contentReference[oaicite:13]{index=13}
    meta_chunks = []
    chunk_size = 200
    for i in range(0, len(item_ids), chunk_size):
        chunk = item_ids[i:i + chunk_size]
        try:
            meta = client.get_items_by_ids(chunk, fields=fields, enable_images=True, enable_userdata=False)
            if not meta.empty:
                meta_chunks.append(meta)
        except Exception:
            continue
    items_meta = pd.concat(meta_chunks, ignore_index=True) if meta_chunks else pd.DataFrame(columns=["ItemId"])

    typer.echo("Building per-user reports...")
    for user_id, df_u in df.groupby("UserId"):
        user_name = user_map.get(user_id, f"unknown_{user_id[:8]}")
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in user_name).strip("_") or "user"
        user_out = out_root / safe_name

        # Compute metrics
        m = compute_metrics_for_user(df_u, items_meta)

        # Optional poster for top title
        poster = None
        if cfg.include_posters and m.get("top_item_id"):
            # Find tag from metadata
            tag = None
            if not items_meta.empty:
                row = items_meta[items_meta["ItemId"] == m["top_item_id"]]
                if not row.empty:
                    tag = row.iloc[0].get("PrimaryImageTag")
            poster = client.fetch_primary_image(m["top_item_id"], tag)

        pages = render_pages(user_out, user_name, cfg.year, m, poster)

        pdf_path = user_out / f"emby_wrapped_{cfg.year}.pdf"
        write_pdf(pdf_path, pages)

        typer.echo(f"✅ {user_name}: {len(pages)} pages -> {user_out}")

    typer.echo("Done.")


if __name__ == "__main__":
    app()

