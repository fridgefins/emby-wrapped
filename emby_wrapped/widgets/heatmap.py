# # Heatmap widget: Day-of-week x Hour viewing intensity.

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
        "axes.edgecolor": (1, 1, 1, 0.25),
        "font.family": "DejaVu Sans",
    })


@register_widget("heatmap")
class HeatmapWidget(BaseWidget):
    key = "heatmap"

    def render(self, report, assets) -> WidgetResult:
        chart_cfg = self.config.get("chart", {}) if isinstance(self.config.get("chart"), dict) else {}
        cmap = str(chart_cfg.get("cmap", "magma"))

        _apply_dark_axes(self.ctx)
        hm = report.heatmap.copy() / 3600.0  # # hours

        fig, ax = plt.subplots(figsize=(11, 4.8))
        im = ax.imshow(hm.values, aspect="auto", interpolation="nearest", cmap=cmap)

        ax.set_title("Viewing Heatmap (Hours)")
        ax.set_yticks(list(range(len(hm.index))))
        ax.set_yticklabels(list(hm.index))
        ax.set_xticks(list(range(24)))
        ax.set_xticklabels([format_hour_12h(h) for h in range(24)], rotation=45, ha="right")

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Hours")

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        url = assets.write_bytes("charts/heatmap.png", buf.getvalue())

        html = f"""
<div class="card wide">
  <div class="sub">Viewing Heatmap</div>
  <div style="margin-top:10px;"><img class="chart" src="{url}" alt="heatmap"></div>
</div>
"""
        return WidgetResult(html=html)
