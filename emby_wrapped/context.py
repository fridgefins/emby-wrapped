# # Shared run context to avoid generator<->widget circular imports.

from __future__ import annotations

import dataclasses

from .config import Config
from .layout import Layout
from .themes import Theme


@dataclasses.dataclass
class RunContext:
    cfg: Config
    layout: Layout
    theme: Theme
