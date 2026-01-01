from __future__ import annotations

from typing import Iterable

import pandas as pd
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .console import ConsoleOptions, make_console


def _safe_str(v: object) -> str:
    if v is None:
        return ""
    return str(v)


def render_debug_sample(
    df: pd.DataFrame,
    *,
    width: int | None = None,
    no_color: bool = False,
    sample_rows: int = 25,
    top_paused_rows: int = 10,
) -> None:
    console = make_console(ConsoleOptions(width=width, no_color=no_color))

    # Header summary
    rows = len(df)
    users = df["UserId"].nunique() if "UserId" in df.columns else 0
    items = df["ItemId"].nunique() if "ItemId" in df.columns else 0

    console.print(
        Panel.fit(
            f"[k]Debug Sample[/k]\n"
            f"[info]Rows[/info]: {rows:,}  "
            f"[info]Users[/info]: {users:,}  "
            f"[info]Items[/info]: {items:,}",
            border_style="info",
        )
    )

    # Choose columns that matter for sanity checks
    cols = [
        "DateCreated",
        "UserId",
        "ItemId",
        "ItemName",
        "PlayDuration",
        "PauseDuration",
        "Seconds",
        "DeviceName",
        "ClientName",
        "PlaybackMethod",
    ]
    cols = [c for c in cols if c in df.columns]

    # Build a fixed-width, no-wrap table.
    # IMPORTANT: each column has max_width and no_wrap=True so the line never wraps.
    t = Table(
        title="Sample rows",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        expand=False,
        padding=(0, 1),
    )

    # Column widths tuned for a ~140-col terminal; adjust if you prefer.
    # Truncation is handled via overflow="ellipsis".
    col_specs = {
        "DateCreated": dict(justify="left",  max_width=19),
        "UserId":      dict(justify="left",  max_width=12),
        "ItemId":      dict(justify="right", max_width=10),
        "ItemName":    dict(justify="left",  max_width=34),
        "PlayDuration":dict(justify="right", max_width=10),
        "PauseDuration":dict(justify="right", max_width=10),
        "Seconds":     dict(justify="right", max_width=10),
        "DeviceName":  dict(justify="left",  max_width=22),
        "ClientName":  dict(justify="left",  max_width=10),
        "PlaybackMethod":dict(justify="left", max_width=18),
    }

    for c in cols:
        spec = col_specs.get(c, dict(justify="left", max_width=16))
        t.add_column(
            c,
            no_wrap=True,
            overflow="ellipsis",
            **spec,
        )

    # Add rows
    view = df[cols].head(sample_rows).copy()

    # Make DateCreated compact
    if "DateCreated" in view.columns:
        view["DateCreated"] = view["DateCreated"].astype(str).str.slice(0, 19)

    for _, r in view.iterrows():
        t.add_row(*[_safe_str(r.get(c)) for c in cols])

    console.print(t)

    # Top paused sessions table
    if "PauseDuration" in df.columns:
        paused = pd.to_numeric(df["PauseDuration"], errors="coerce").fillna(0.0)
        n_paused = int((paused > 0).sum())

        console.print(
            Panel.fit(
                f"[info]Rows with pause > 0[/info]: {n_paused:,} ({(n_paused / max(rows, 1)):.1%})",
                border_style="warn" if n_paused else "ok",
            )
        )

        if n_paused:
            t2 = Table(
                title="Top paused sessions (by PauseDuration)",
                box=box.SIMPLE_HEAVY,
                show_lines=False,
                expand=False,
                padding=(0, 1),
            )
            for c in cols:
                spec = col_specs.get(c, dict(justify="left", max_width=16))
                t2.add_column(c, no_wrap=True, overflow="ellipsis", **spec)

            top = df.assign(_Pause=paused).sort_values("_Pause", ascending=False)[cols].head(top_paused_rows).copy()
            if "DateCreated" in top.columns:
                top["DateCreated"] = top["DateCreated"].astype(str).str.slice(0, 19)

            for _, r in top.iterrows():
                t2.add_row(*[_safe_str(r.get(c)) for c in cols])

            console.print(t2)
