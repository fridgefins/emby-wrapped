# # Devices widget: top devices/clients/methods.

from __future__ import annotations

from .base import BaseWidget, WidgetResult
from .registry import register_widget
from ..util import format_duration_dhms


def _table(series, limit: int):
    if series is None or len(series) == 0:
        return "<tr><td>—</td><td class='mono'>—</td></tr>"
    rows = []
    for k, v in series.head(limit).items():
        name = (str(k) if k else "Unknown").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        rows.append(f"<tr><td>{name}</td><td class='mono'>{format_duration_dhms(float(v))}</td></tr>")
    return "\n".join(rows)


@register_widget("devices")
class DevicesWidget(BaseWidget):
    key = "devices"

    def render(self, report, assets) -> WidgetResult:
        limit = int(self.config.get("limit", 10))
        html = f"""
<div class="card wide">
  <div class="sub">Devices & Clients</div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px;">
    <div>
      <div class="badge">Devices</div>
      <table style="margin-top:8px;"><thead><tr><th>Name</th><th class="mono">Time</th></tr></thead><tbody>
        {_table(report.device_seconds, limit)}
      </tbody></table>
    </div>
    <div>
      <div class="badge">Clients</div>
      <table style="margin-top:8px;"><thead><tr><th>Name</th><th class="mono">Time</th></tr></thead><tbody>
        {_table(report.client_seconds, limit)}
      </tbody></table>
    </div>
    <div>
      <div class="badge">Playback Method</div>
      <table style="margin-top:8px;"><thead><tr><th>Name</th><th class="mono">Time</th></tr></thead><tbody>
        {_table(report.method_seconds, limit)}
      </tbody></table>
    </div>
  </div>
</div>
"""
        return WidgetResult(html=html)
