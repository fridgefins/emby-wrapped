# # Widget registry: drop in widgets without touching core engine.

from __future__ import annotations

from typing import Callable, Dict, TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .base import BaseWidget

_WIDGETS: Dict[str, Type["BaseWidget"]] = {}


def register_widget(key: str) -> Callable[[Type["BaseWidget"]], Type["BaseWidget"]]:
    def deco(cls: Type["BaseWidget"]) -> Type["BaseWidget"]:
        _WIDGETS[key] = cls
        return cls
    return deco


def get_widget(key: str) -> Type["BaseWidget"]:
    if key not in _WIDGETS:
        raise KeyError(f"Unknown widget: {key}. Available: {sorted(_WIDGETS.keys())}")
    return _WIDGETS[key]
