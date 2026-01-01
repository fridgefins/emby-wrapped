# # Generator interface.

from __future__ import annotations

from pathlib import Path

from ..context import RunContext
from ..metrics.report_builder import UserReport
from ..output.assets import AssetWriter


class BaseGenerator:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def render_user(self, report: UserReport, out_dir: Path, assets: AssetWriter) -> None:
        raise NotImplementedError
