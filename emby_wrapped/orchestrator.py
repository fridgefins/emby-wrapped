# # Orchestrates: load playback DB, fetch user map + item meta (optional), build per-user reports,
# # then call generator to write output.

from __future__ import annotations

from pathlib import Path
from typing import Dict

from . import widgets
from . import generators
from .config import Config
from .layout import Layout
from .themes import pick_theme
from .data.playback_db import load_playback_df
from .data.emby_api import EmbyClient, EmbyConn
from .data.enrich import enrich_items_meta
from .metrics.report_builder import build_user_report
from .context import RunContext
from .generators.registry import get_generator
from .output.assets import AssetWriter

def run(cfg: Config, layout: Layout) -> None:
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # # 1) Load playback data
    df = load_playback_df(cfg.playback_db, cfg.year, cfg.timezone)
    if df.empty:
        return

    # # 2) Optional Emby enrichment
    client = None
    user_map: Dict[str, str] = {}
    items_meta = None

    if cfg.server_url and cfg.base_path and cfg.api_key:
        conn = EmbyConn(
            server_url=cfg.server_url,
            base_path=cfg.base_path,
            api_key=cfg.api_key,
            timeout_seconds=cfg.http_timeout_seconds,
        )
        client = EmbyClient(conn)

        # # Users
        try:
            users_df = client.fetch_users()
            user_map = {str(r["UserId"]): str(r["UserName"]) for _, r in users_df.iterrows()}
        except Exception:
            user_map = {}

        # # Items meta
        try:
            item_ids = sorted(df["ItemId"].unique().tolist())
            items_meta = client.fetch_items_meta(item_ids)
        except Exception:
            items_meta = None

    items_meta = enrich_items_meta(df, items_meta)

    # # 3) Generator context (theme is per-user; ctx will be updated per user)
    gen_cls = get_generator(layout.generator)

    # # 4) Per-user generation
    for user_id in sorted(df["UserId"].unique().tolist()):
        user_name = user_map.get(str(user_id), str(user_id))

        # # Theme selection per user
        theme = pick_theme(seed=str(user_id), palette=layout.theme.palette)

        ctx = RunContext(cfg=cfg, layout=layout, theme=theme)
        gen = gen_cls(ctx)

        user_out = out_root / _safe_name(user_name)
        user_out.mkdir(parents=True, exist_ok=True)
        assets = AssetWriter(root=user_out)

        df_u = df[df["UserId"] == str(user_id)].copy()
        report = build_user_report(
            year=cfg.year,
            user_id=str(user_id),
            user_name=user_name,
            df_user=df_u,
            items_meta=items_meta,
            max_top=cfg.max_top_titles,
        )

        # # Optionally: let widgets request posters via Emby; you can store client on ctx later if desired.
        gen.render_user(report, user_out, assets)


def _safe_name(s: str) -> str:
    out = "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_")).strip()
    return out or "user"
