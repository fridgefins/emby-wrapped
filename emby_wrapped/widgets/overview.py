# # Overview widget: KPIs and totals.

from __future__ import annotations

from .base import BaseWidget, WidgetResult
from .registry import register_widget
from ..util import format_duration_dhms


@register_widget("overview")
class OverviewWidget(BaseWidget):
    key = "overview"

    def render(self, report, assets) -> WidgetResult:
        html = f"""
<div class="card third">
  <div><div class="sub">Total Playtime</div><div style="font-size:18px;font-weight:800" class="mono">{format_duration_dhms(report.total_seconds)}</div></div>
  <div style="margin-top:10px"><div class="sub">Plays</div><div style="font-size:18px;font-weight:800" class="mono">{report.plays}</div></div>
  <div style="margin-top:10px"><div class="sub">Distinct Titles</div><div style="font-size:18px;font-weight:800" class="mono">{report.distinct_titles}</div></div>
</div>
"""
        return WidgetResult(html=html)

