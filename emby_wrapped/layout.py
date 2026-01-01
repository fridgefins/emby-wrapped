# # Layout loader: supports both list-form and dict-form widget configs.

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class ThemeSpec:
    palette: str = "auto"  # # auto | Sunset | Neon | Aurora | Citrus | Grape


@dataclasses.dataclass
class WidgetSpec:
    key: str
    enabled: bool = True
    config: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Layout:
    generator: str = "html_v1"
    theme: ThemeSpec = dataclasses.field(default_factory=ThemeSpec)
    widgets: List[WidgetSpec] = dataclasses.field(default_factory=list)


def _coerce_widgets(w: Any) -> List[WidgetSpec]:
    # # List form: [{"key": "...", "enabled": true, ...}, ...]
    if isinstance(w, list):
        out: List[WidgetSpec] = []
        for entry in w:
            if not isinstance(entry, dict) or "key" not in entry:
                continue
            key = str(entry["key"])
            enabled = bool(entry.get("enabled", True))
            cfg = {k: v for k, v in entry.items() if k not in {"key", "enabled"}}
            out.append(WidgetSpec(key=key, enabled=enabled, config=cfg))
        return out

    # # Dict form: {"playtime-by-hour": {...}, "top-titles": {...}}
    if isinstance(w, dict):
        out = []
        for key, cfg in w.items():
            cfg = cfg if isinstance(cfg, dict) else {}
            enabled = bool(cfg.get("enabled", True))
            cfg2 = {k: v for k, v in cfg.items() if k != "enabled"}
            out.append(WidgetSpec(key=str(key), enabled=enabled, config=cfg2))
        return out

    return []


def load_layout(path: Path) -> Layout:
    if not path.exists():
        # # Default: sensible baseline widgets
        return Layout(
            generator="html_v1",
            theme=ThemeSpec(palette="auto"),
            widgets=[
                WidgetSpec("overview"),
                WidgetSpec("playtime-split"),
                WidgetSpec("playtime-by-month"),
                WidgetSpec("playtime-by-hour"),
                WidgetSpec("playtime-by-dow"),
                WidgetSpec("top-titles"),
            ],
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    gen = str(data.get("generator", "html_v1"))
    theme_raw = data.get("theme") or {}
    theme = ThemeSpec(palette=str(theme_raw.get("palette", "auto")))
    widgets = _coerce_widgets(data.get("widgets"))

    return Layout(generator=gen, theme=theme, widgets=widgets)
