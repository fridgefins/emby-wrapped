# # Bar/line chart: watch time by hour.

from __future__ import annotations

import io

import matplotlib.pyplot as plt

from .base import BaseWidget, WidgetResult
from .registry import register_widget
from ..util import format_hour_12h


def _apply_dark_axes(theme):
    bg = tuple(c / 255.0 for c in theme.bg1)
    plt.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "text.color": (1, 1, 1, 0.92),
        "axes.labelcolor": (1, 1, 1, 0.92),
        "xtick.color": (1, 1, 1, 0.92),
        "ytick.color": (1, 1, 1, 0.92),
        "grid.color": (1, 1, 1, 0.14),
        "axes.edgecolor": (1, 1, 1, 0.25),
        "font.family": "DejaVu Sans",
    })


def _resolve_color(ctx, name: str):
    if name == "accent2":
        return tuple(c / 255.0 for c in ctx.theme.accent2)
    if name == "theme" or name == "accent":
        return tuple(c / 255.0 for c in ctx.theme.accent)
    return tuple(c / 255.0 for c in ctx.theme.accent)


@register_widget("playtime-by-hour")
class PlaytimeByHourWidget(BaseWidget):
    key = "playtime-by-hour"

    def render(self, report, assets) -> WidgetResult:
        chart_cfg = self.config.get("chart", {}) if isinstance(self.config.get("chart"), dict) else {}
        chart_type = str(chart_cfg.get("type", "bar")).lower()
        color_name = str(chart_cfg.get("color", "accent"))

        _apply_dark_axes(self.ctx)
        color = _resolve_color(self.ctx, color_name)

        hours = report.hour_seconds.index
        vals = report.hour_seconds.values / 3600.0

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.grid(True, axis="y")

        if chart_type == "line":
            ax.plot(hours, vals, marker="o", linewidth=2.5, color=color, alpha=0.95)
        else:
            ax.bar(hours, vals, color=color, alpha=0.90)

        ax.set_title("Watch Time by Hour")
        ax.set_ylabel("Hours")
        ax.set_xlabel("Hour")
        ax.set_xticks(list(range(24)))
        ax.set_xticklabels([format_hour_12h(h) for h in range(24)], rotation=45, ha="right")

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        url = assets.write_bytes("charts/hour.png", buf.getvalue())

        html = f"""
<div class="card">
  <div class="sub">Watch Time by Hour</div>
  <div style="margin-top:10px;"><img class="chart" src="{url}" alt="hour"></div>
</div>
"""
        return WidgetResult(html=html)
