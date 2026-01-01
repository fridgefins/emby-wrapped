# # HTML shell: wraps rendered widgets in a consistent page.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..themes import Theme


def _rgb(theme_tuple) -> str:
    return f"{theme_tuple[0]},{theme_tuple[1]},{theme_tuple[2]}"


def build_css(theme: Theme) -> str:
    # # Keep this centralized so generators/widgets don’t duplicate CSS
    return f"""
:root{{
  --bg1: rgb({_rgb(theme.bg1)});
  --bg2: rgb({_rgb(theme.bg2)});
  --accent: rgb({_rgb(theme.accent)});
  --accent2: rgb({_rgb(theme.accent2)});
  --text: rgba(255,255,255,.92);
  --muted: rgba(255,255,255,.72);
  --card: rgba(255,255,255,.08);
  --stroke: rgba(255,255,255,.14);
  --shadow: rgba(0,0,0,.35);
}}
html,body{{
  height:100%;
  margin:0;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  color: var(--text);
  background:
    radial-gradient(1200px 600px at 15% 10%, rgba(255,255,255,.08), transparent 60%),
    radial-gradient(900px 500px at 85% 25%, rgba(255,255,255,.06), transparent 55%),
    linear-gradient(145deg, var(--bg1), var(--bg2));
}}
.container{{ max-width:1120px; margin:0 auto; padding:28px 18px 60px; }}
.hero{{
  padding:22px;
  border:1px solid var(--stroke);
  background: linear-gradient(145deg, var(--card), rgba(255,255,255,.04));
  border-radius:18px;
  box-shadow:0 18px 40px var(--shadow);
}}
.heroTop{{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; flex-wrap:wrap; }}
h1{{ margin:0; font-size:34px; letter-spacing:-0.02em; }}
.sub{{ color:var(--muted); font-size:14px; margin-top:4px; }}
.badge{{ display:inline-block; padding:5px 10px; border:1px solid rgba(255,255,255,.16); border-radius:999px; color:var(--muted); font-size:12px; }}
.grid{{ display:grid; grid-template-columns:repeat(12,1fr); gap:14px; margin-top:14px; }}
.card{{ grid-column: span 6; border-radius:18px; border:1px solid var(--stroke);
        background: linear-gradient(145deg, var(--card), rgba(0,0,0,.12)); padding:16px; }}
.card.wide{{ grid-column: span 12; }}
.card.third{{ grid-column: span 4; }}
img.chart{{ width:100%; border-radius:14px; border:1px solid rgba(255,255,255,.12); box-shadow:0 12px 30px rgba(0,0,0,.30); }}
table{{ width:100%; border-collapse:collapse; }}
td,th{{ padding:8px 6px; border-bottom:1px solid rgba(255,255,255,.10); text-align:left; }}
th{{ color:var(--muted); font-weight:650; font-size:12px; }}
.mono{{ font-variant-numeric: tabular-nums; }}
"""

def wrap_page(title: str, subtitle: str, theme_name: str, body_html: str, theme: Theme) -> str:
    css = build_css(theme)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <div class="heroTop">
        <div>
          <h1>{title}</h1>
          <div class="sub">{subtitle}</div>
        </div>
        <div class="badge">theme: {theme_name}</div>
      </div>
    </div>
    <div class="grid">
      {body_html}
    </div>
  </div>
</body>
</html>
"""

