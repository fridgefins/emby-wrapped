# # Build a per-user report object: metrics + distributions used by widgets.

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..util import DOW_ORDER, format_duration_dhms, format_hour_12h, norm_str


@dataclasses.dataclass
class UserReport:
    user_id: str
    user_name: str
    year: int

    # # Core totals
    plays: int
    distinct_titles: int
    total_seconds: float

    # # Split seconds
    movie_seconds: float
    tvshow_seconds: float
    tvch_seconds: float
    other_seconds: float

    # # Peak
    peak_day: str
    peak_hour: int
    peak_hour_label: str

    # # Top lists
    top_movies: List[Dict[str, Any]]
    top_tvshows: List[Dict[str, Any]]
    top_tvchannels: List[Dict[str, Any]]

    # # Distributions
    hour_seconds: pd.Series
    dow_seconds: pd.Series
    month_seconds: pd.Series

    # # Optional breakdowns
    device_seconds: pd.Series
    client_seconds: pd.Series
    method_seconds: pd.Series

    # # Heatmap (dow x hour)
    heatmap: pd.DataFrame


def _classify(emby_type: Any, fallback: Any) -> str:
    t = (norm_str(emby_type) or norm_str(fallback)).lower()
    if not t:
        return "other"
    if t == "movie":
        return "movie"
    if t == "tvchannel":
        return "tv_channel"
    if t in {"episode", "series", "season"}:
        return "tv_show"
    return "other"


def _build_display_title(df_items: pd.DataFrame) -> pd.Series:
    # # For tv_show: SeriesName else Name
    name = df_items["Name"].fillna(df_items.get("ItemName")).fillna("").astype("string")
    series = df_items.get("SeriesName")
    if series is None:
        series = pd.Series([None] * len(df_items), index=df_items.index)

    series = series.fillna("").astype("string")
    out = name.copy()

    tv_mask = df_items["Class"].eq("tv_show")
    out[tv_mask] = series[tv_mask]
    out = out.mask(out.str.len().eq(0), name)
    return out


def _top_list(df_items: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
    if df_items.empty:
        return []
    s = (
        df_items.groupby("DisplayTitle")["Seconds"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    return [{"title": k, "seconds": float(v), "time": format_duration_dhms(float(v))} for k, v in s.items()]


def build_user_report(
    year: int,
    user_id: str,
    user_name: str,
    df_user: pd.DataFrame,
    items_meta: pd.DataFrame,
    max_top: int,
) -> UserReport:
    # # Aggregate per item
    by_item = df_user.groupby("ItemId", as_index=False)["Seconds"].sum()

    merged = by_item.merge(items_meta, on="ItemId", how="left")

    # # Fallback columns from playback table
    fallback = df_user[["ItemId", "ItemType", "ItemName"]].drop_duplicates(subset=["ItemId"])
    merged = merged.merge(fallback, on="ItemId", how="left", suffixes=("", "_fb"))

    # # Effective type/name
    merged["TypeEff"] = merged["Type"].combine_first(merged["ItemType"])
    merged["Name"] = merged["Name"].combine_first(merged["ItemName"])

    merged["Class"] = merged.apply(lambda r: _classify(r.get("TypeEff"), r.get("ItemType")), axis=1)
    merged["DisplayTitle"] = _build_display_title(merged)

    plays = int(len(df_user))
    distinct_titles = int(df_user["ItemId"].nunique())
    total_seconds = float(df_user["Seconds"].sum())

    class_sum = merged.groupby("Class")["Seconds"].sum().to_dict()
    movie_seconds = float(class_sum.get("movie", 0.0))
    tvshow_seconds = float(class_sum.get("tv_show", 0.0))
    tvch_seconds = float(class_sum.get("tv_channel", 0.0))
    other_seconds = float(class_sum.get("other", 0.0))

    # # Peak day/hour by seconds
    dow_sum = df_user.groupby("DayOfWeek")["Seconds"].sum().reindex(DOW_ORDER, fill_value=0.0)
    peak_day = str(dow_sum.idxmax()) if len(dow_sum) else "—"

    hour_sum = df_user.groupby("Hour")["Seconds"].sum().reindex(range(24), fill_value=0.0)
    peak_hour = int(hour_sum.idxmax()) if len(hour_sum) else 0
    peak_hour_label = format_hour_12h(peak_hour)

    # # Distributions
    month_sum = df_user.groupby("Month")["Seconds"].sum().sort_index()

    # # Breakdowns
    dev_sum = df_user.groupby("DeviceName")["Seconds"].sum().sort_values(ascending=False)
    client_sum = df_user.groupby("ClientName")["Seconds"].sum().sort_values(ascending=False)
    method_sum = df_user.groupby("PlaybackMethod")["Seconds"].sum().sort_values(ascending=False)

    # # Heatmap: DOW x Hour
    hm = (
        df_user.pivot_table(
            index="DayOfWeek",
            columns="Hour",
            values="Seconds",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=DOW_ORDER, columns=list(range(24)), fill_value=0.0)
    )

    return UserReport(
        user_id=str(user_id),
        user_name=str(user_name),
        year=int(year),

        plays=plays,
        distinct_titles=distinct_titles,
        total_seconds=total_seconds,

        movie_seconds=movie_seconds,
        tvshow_seconds=tvshow_seconds,
        tvch_seconds=tvch_seconds,
        other_seconds=other_seconds,

        peak_day=peak_day,
        peak_hour=peak_hour,
        peak_hour_label=peak_hour_label,

        top_movies=_top_list(merged[merged["Class"].eq("movie")], max_top),
        top_tvshows=_top_list(merged[merged["Class"].eq("tv_show")], max_top),
        top_tvchannels=_top_list(merged[merged["Class"].eq("tv_channel")], max_top),

        hour_seconds=hour_sum,
        dow_seconds=dow_sum,
        month_seconds=month_sum,

        device_seconds=dev_sum,
        client_seconds=client_sum,
        method_seconds=method_sum,

        heatmap=hm,
    )

