# # Widget interface: returns HTML fragments (and can emit assets via AssetWriter).

from __future__ import annotations

import dataclasses
from typing import Any, Dict

from ..context import RunContext
from ..metrics.report_builder import UserReport
from ..output.assets import AssetWriter


@dataclasses.dataclass
class WidgetResult:
    html: str


class BaseWidget:
    key: str = "base"

    def __init__(self, ctx: RunContext, config: Dict[str, Any]):
        self.ctx = ctx
        self.config = config

    def render(self, report: UserReport, assets: AssetWriter) -> WidgetResult:
        raise NotImplementedError
