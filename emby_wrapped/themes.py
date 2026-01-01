# # Theme palettes + deterministic selection.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Theme:
    name: str
    accent: Tuple[int, int, int]
    accent2: Tuple[int, int, int]
    bg1: Tuple[int, int, int]
    bg2: Tuple[int, int, int]


PALETTES: Dict[str, Theme] = {
    "Sunset": Theme("Sunset", accent=(255, 140, 66),  accent2=(255, 72, 164), bg1=(14, 10, 10), bg2=(42, 16, 28)),
    "Neon":   Theme("Neon",   accent=(0, 224, 255),   accent2=(255, 77, 240), bg1=(8, 10, 18),  bg2=(20, 10, 35)),
    "Aurora": Theme("Aurora", accent=(78, 255, 168),  accent2=(88, 171, 255), bg1=(8, 14, 12),  bg2=(8, 20, 32)),
    "Citrus": Theme("Citrus", accent=(255, 208, 66),  accent2=(80, 255, 140), bg1=(12, 12, 8),  bg2=(18, 24, 10)),
    "Grape":  Theme("Grape",  accent=(180, 120, 255), accent2=(255, 95, 160), bg1=(10, 8, 16),  bg2=(24, 10, 20)),
}


def pick_theme(seed: str, palette: str = "auto") -> Theme:
    if palette and palette != "auto" and palette in PALETTES:
        return PALETTES[palette]

    names = list(PALETTES.keys())
    h = 0
    for ch in seed:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return PALETTES[names[h % len(names)]]
