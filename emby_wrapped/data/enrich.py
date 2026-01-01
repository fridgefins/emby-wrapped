# # Join playback rows to meta, providing fallbacks when meta is missing.

from __future__ import annotations

import pandas as pd

from ..util import norm_str


def enrich_items_meta(df_playback: pd.DataFrame, items_meta: pd.DataFrame) -> pd.DataFrame:
    # # Ensure item meta has string ItemId
    if items_meta is None or items_meta.empty:
        return pd.DataFrame(columns=["ItemId", "Name", "Type", "SeriesName", "SeriesId", "Genres", "PrimaryImageTag"])

    meta = items_meta.copy()
    meta["ItemId"] = meta["ItemId"].astype(str)

    # # If any obvious missing fields, normalize
    for col in ["Name", "Type", "SeriesName", "SeriesId", "PrimaryImageTag"]:
        if col not in meta.columns:
            meta[col] = None
    if "Genres" not in meta.columns:
        meta["Genres"] = [[] for _ in range(len(meta))]

    return meta
