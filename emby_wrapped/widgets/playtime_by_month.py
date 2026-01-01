# # Line chart: watch time by month.

from __future__ import annotations

import io

import matplotlib.pyplot as plt

from .base import BaseWidget, WidgetResult
from .registry import register_widget


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


@register_widget("playtime-by-month")
class PlaytimeByMonthWidget(BaseWidget):
    key = "playtime-by-month"

    def render(self, report, assets) -> WidgetResult:
        _apply_dark_axes(self.ctx)

        color = tuple(c / 255.0 for c in self.ctx.theme.accent)
        labels = list(report.month_seconds.index)
        vals = report.month_seconds.values / 3600.0

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.grid(True, axis="y")
        ax.plot(range(len(labels)), vals, marker="o", linewidth=2.5, color=color, alpha=0.95)
        ax.set_title("Watch Time by Month")
        ax.set_ylabel("Hours")
        ax.set_xlabel("Month")
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels, rotation=45, ha="right")

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        url = assets.write_bytes("charts/month.png", buf.getvalue())

        html = f"""
<div class="card">
  <div class="sub">Watch Time by Month</div>
  <div style="margin-top:10px;"><img class="chart" src="{url}" alt="month"></div>
</div>
"""
        return WidgetResult(html=html)
