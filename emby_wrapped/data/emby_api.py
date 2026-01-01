# # Minimal Emby API client for users, metadata, posters.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


@dataclass
class EmbyConn:
    server_url: str
    base_path: str
    api_key: str
    timeout_seconds: int = 25

    @property
    def base(self) -> str:
        return self.server_url.rstrip("/") + self.base_path.rstrip("/")


class EmbyClient:
    def __init__(self, conn: EmbyConn):
        self.conn = conn
        self.session = requests.Session()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.conn.base + path
        params = dict(params or {})
        if self.conn.api_key:
            params["api_key"] = self.conn.api_key
        r = self.session.get(url, params=params, timeout=self.conn.timeout_seconds)
        r.raise_for_status()
        return r.json()

    def fetch_users(self) -> pd.DataFrame:
        data = self._get("/Users")
        rows = [{"UserId": u.get("Id"), "UserName": u.get("Name")} for u in data]
        return pd.DataFrame(rows)

    def fetch_items_meta(self, item_ids: List[str]) -> pd.DataFrame:
        out: List[pd.DataFrame] = []
        chunk_size = 200

        for i in range(0, len(item_ids), chunk_size):
            chunk = item_ids[i:i + chunk_size]
            data = self._get(
                "/Items",
                params={
                    "Ids": ",".join(map(str, chunk)),
                    "Fields": "Genres,PrimaryImageTag,SeriesName,SeriesId,Type,Name",
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
                    "PrimaryImageTag": (it.get("ImageTags", {}) or {}).get("Primary") or it.get("PrimaryImageTag"),
                })
            if rows:
                out.append(pd.DataFrame(rows))

        if not out:
            return pd.DataFrame(columns=["ItemId", "Name", "Type", "SeriesName", "SeriesId", "Genres", "PrimaryImageTag"])

        meta = pd.concat(out, ignore_index=True)
        meta["ItemId"] = meta["ItemId"].astype(str)
        return meta

    def download_primary_image(self, item_id: str, tag: Optional[str], max_width: int) -> Optional[bytes]:
        params: Dict[str, Any] = {"maxWidth": str(max_width), "quality": "90"}
        if self.conn.api_key:
            params["api_key"] = self.conn.api_key
        if tag:
            params["tag"] = tag

        url = self.conn.base + f"/Items/{item_id}/Images/Primary"
        r = self.session.get(url, params=params, timeout=self.conn.timeout_seconds)
        if r.status_code != 200 or not r.content:
            return None
        return r.content
