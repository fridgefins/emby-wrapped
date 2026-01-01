# # Split widget: Movies / TV Shows / TV Channels (plus Other).

from __future__ import annotations

from .base import BaseWidget, WidgetResult
from .registry import register_widget
from ..util import format_duration_dhms


@register_widget("playtime-split")
class PlaytimeSplitWidget(BaseWidget):
    key = "playtime-split"

    def render(self, report, assets) -> WidgetResult:
        html = f"""
<div class="card" style="grid-column: span 8;">
  <div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;">
    <div class="sub">Playtime Split</div>
    <div class="sub">Movies / TV / Live</div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px;">
    <div style="border:1px solid rgba(255,255,255,.10);border-radius:14px;padding:12px;background:linear-gradient(145deg,rgba(255,255,255,.06),rgba(0,0,0,.12));">
      <div class="sub">Movies</div>
      <div style="font-weight:850" class="mono">{format_duration_dhms(report.movie_seconds)}</div>
    </div>
    <div style="border:1px solid rgba(255,255,255,.10);border-radius:14px;padding:12px;background:linear-gradient(145deg,rgba(255,255,255,.06),rgba(0,0,0,.12));">
      <div class="sub">TV Shows</div>
      <div style="font-weight:850" class="mono">{format_duration_dhms(report.tvshow_seconds)}</div>
    </div>
    <div style="border:1px solid rgba(255,255,255,.10);border-radius:14px;padding:12px;background:linear-gradient(145deg,rgba(255,255,255,.06),rgba(0,0,0,.12));">
      <div class="sub">TV Channels</div>
      <div style="font-weight:850" class="mono">{format_duration_dhms(report.tvch_seconds)}</div>
    </div>
  </div>

  <div class="sub" style="margin-top:10px;">Other: <b class="mono">{format_duration_dhms(report.other_seconds)}</b></div>
</div>
"""
        return WidgetResult(html=html)
