# # Asset writing for a user output folder.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AssetWriter:
    root: Path

    def write_bytes(self, rel_path: str, data: bytes) -> str:
        # # Returns a browser-friendly relative URL
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return rel_path.replace("\\", "/")
