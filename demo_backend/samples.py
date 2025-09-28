from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Optional

from .config import DemoSettings


@dataclass(slots=True)
class DemoSample:
    sample_id: str
    title: str
    description: str
    image_path: str
    overlay_path: Optional[str]
    cam_paths: Dict[str, str]
    findings: List[Dict[str, object]]

    def to_summary(self) -> Dict[str, object]:
        return {
            "id": self.sample_id,
            "title": self.title,
            "description": self.description,
            "image_path": self.image_path,
            "overlay_path": self.overlay_path,
            "cam_paths": self.cam_paths,
            "findings": self.findings,
        }


class SampleStore:
    def __init__(self, settings: DemoSettings) -> None:
        self.settings = settings
        self._items: Dict[str, DemoSample] = {}
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        manifest_path = self.settings.samples_manifest
        if not manifest_path.exists():
            self._items = {}
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples: Dict[str, DemoSample] = {}
        for entry in raw:
            sample = DemoSample(
                sample_id=entry["id"],
                title=entry.get("title", entry["id"]),
                description=entry.get("description", ""),
                image_path=entry.get("image_path", ""),
                overlay_path=entry.get("overlay_path"),
                cam_paths=entry.get("cam_paths", {}),
                findings=entry.get("findings", []),
            )
            samples[sample.sample_id] = sample
        self._items = samples

    # ------------------------------------------------------------------
    def list(self) -> List[DemoSample]:
        return list(self._items.values())

    # ------------------------------------------------------------------
    def get(self, sample_id: str) -> Optional[DemoSample]:
        return self._items.get(sample_id)

    # ------------------------------------------------------------------
    def to_response(self) -> List[Dict[str, object]]:
        return [sample.to_summary() for sample in self.list()]
