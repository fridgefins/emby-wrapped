# # HTML generator: stitches widget HTML into a page.

from __future__ import annotations

from pathlib import Path

from .base import BaseGenerator
from .registry import register_generator
from ..output.assets import AssetWriter
from ..output.html_shell import wrap_page
from ..widgets.registry import get_widget


@register_generator("html_v1")
class HtmlV1Generator(BaseGenerator):
    def render_user(self, report, out_dir: Path, assets: AssetWriter) -> None:
        # # Render widgets in layout order
        chunks = []
        for spec in self.ctx.layout.widgets:
            if not spec.enabled:
                continue
            w_cls = get_widget(spec.key)
            widget = w_cls(self.ctx, spec.config)
            res = widget.render(report, assets)
            chunks.append(res.html)

        title = f"Emby Wrapped {report.year}"
        subtitle = f"for <b>{report.user_name}</b> • peak hour: <b>{report.peak_hour_label}</b> • peak day: <b>{report.peak_day}</b>"
        html = wrap_page(title=title, subtitle=subtitle, theme_name=self.ctx.theme.name, body_html="\n".join(chunks), theme=self.ctx.theme)

        (out_dir / "index.html").write_text(html, encoding="utf-8")
