# # Top titles widget: Movies / TV Shows / TV Channels tables.

from __future__ import annotations

from .base import BaseWidget, WidgetResult
from .registry import register_widget


def _rows(items):
    if not items:
        return "<tr><td>—</td><td class='mono'>—</td></tr>"
    out = []
    for it in items:
        title = (it.get("title") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        time_s = it.get("time") or "0d 00h 00m 00s"
        out.append(f"<tr><td>{title}</td><td class='mono'>{time_s}</td></tr>")
    return "\n".join(out)


@register_widget("top-titles")
class TopTitlesWidget(BaseWidget):
    key = "top-titles"

    def render(self, report, assets) -> WidgetResult:
        # # items-per-type can override report’s computed list size later (we already cap in builder)
        html = f"""
<div class="card wide">
  <div class="sub">Top Titles</div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px;">
    <div>
      <div class="badge">Movies</div>
      <table style="margin-top:8px;"><thead><tr><th>Title</th><th class="mono">Time</th></tr></thead><tbody>
        {_rows(report.top_movies)}
      </tbody></table>
    </div>
    <div>
      <div class="badge">TV Shows</div>
      <table style="margin-top:8px;"><thead><tr><th>Title</th><th class="mono">Time</th></tr></thead><tbody>
        {_rows(report.top_tvshows)}
      </tbody></table>
    </div>
    <div>
      <div class="badge">TV Channels</div>
      <table style="margin-top:8px;"><thead><tr><th>Title</th><th class="mono">Time</th></tr></thead><tbody>
        {_rows(report.top_tvchannels)}
      </tbody></table>
    </div>
  </div>
</div>
"""
        return WidgetResult(html=html)
