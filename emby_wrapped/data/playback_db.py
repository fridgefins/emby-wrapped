# # Load and normalize Playback Reporting DB into a single DataFrame.

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


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
    best_table = ""
    best_score = -1
    best_map: Dict[str, str] = {}

    for t in list_tables(conn):
        cols = table_columns(conn, t)
        cols_set = set(cols)

        cmap: Dict[str, str] = {}
        ok = True
        score = 0

        for key, candidates in REQUIRED_KEYS.items():
            found = None
            for c in candidates:
                if c in cols_set:
                    found = c
                    break
            if not found:
                ok = False
                break
            cmap[key] = found
            score += 5

        if not ok:
            continue

        for out_name, candidates in OPTIONAL_COLS.items():
            for c in candidates:
                if c in cols_set:
                    cmap[out_name] = c
                    score += 1
                    break

        name_l = t.lower()
        if "playback" in name_l:
            score += 2
        if "activity" in name_l:
            score += 1

        if score > best_score:
            best_score = score
            best_table = t
            best_map = cmap

    if not best_table:
        raise RuntimeError("Could not auto-detect playback table in playback_reporting.db")

    return best_table, best_map


def load_playback_df(playback_db: str, year: int, timezone: str) -> pd.DataFrame:
    db_path = Path(playback_db)
    if not db_path.exists():
        raise FileNotFoundError(f"Playback DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        table, cmap = choose_playback_table(conn)

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

        query = f"SELECT {', '.join(select_cols)} FROM {table}"
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    df["UserId"] = df["UserId"].astype(str)
    df["ItemId"] = df["ItemId"].astype(str)

    # # Parse DateCreated robustly
    dc = df["DateCreated"]
    if pd.api.types.is_numeric_dtype(dc):
        s = pd.to_numeric(dc, errors="coerce")
        ms_mask = s > 1_000_000_000_000
        sec = s.copy()
        sec[ms_mask] = sec[ms_mask] / 1000.0
        ts = pd.to_datetime(sec, unit="s", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(dc, utc=True, errors="coerce")
        if ts.isna().mean() > 0.80:
            ts2 = pd.to_datetime(dc, errors="coerce")
            ts = ts2.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")

    df["DateCreated"] = ts.dt.tz_convert(timezone)

    # # Duration as seconds
    df["PlayDuration"] = pd.to_numeric(df["PlayDuration"], errors="coerce").fillna(0.0).astype(float)
    df["Seconds"] = df["PlayDuration"]

    # # Filter to year AFTER timezone conversion
    start = pd.Timestamp(dt.datetime(year, 1, 1), tz=timezone)
    end = pd.Timestamp(dt.datetime(year + 1, 1, 1), tz=timezone)
    df = df[(df["DateCreated"] >= start) & (df["DateCreated"] < end)].copy()

    df["Hour"] = df["DateCreated"].dt.hour.astype(int)
    df["DayOfWeek"] = df["DateCreated"].dt.day_name()
    df["Month"] = df["DateCreated"].dt.to_period("M").astype(str)

    return df
