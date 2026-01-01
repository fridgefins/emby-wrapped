# # Generator registry: lets you plug in new renderers easily.

from __future__ import annotations

from typing import Callable, Dict, TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .base import BaseGenerator

_GENERATORS: Dict[str, Type["BaseGenerator"]] = {}


def register_generator(key: str) -> Callable[[Type["BaseGenerator"]], Type["BaseGenerator"]]:
    def deco(cls: Type["BaseGenerator"]) -> Type["BaseGenerator"]:
        _GENERATORS[key] = cls
        return cls
    return deco


def get_generator(key: str) -> Type["BaseGenerator"]:
    if key not in _GENERATORS:
        raise KeyError(f"Unknown generator: {key}. Available: {sorted(_GENERATORS.keys())}")
    return _GENERATORS[key]
