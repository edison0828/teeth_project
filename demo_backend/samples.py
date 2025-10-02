from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import DemoSettings


@dataclass(slots=True)
class DemoSample:
    sample_id: str
    title: str
    description: str
    image_uri: str
    image_path: Path
    overlay_uri: Optional[str]
    cam_paths: Dict[str, str]

    def to_summary(self) -> Dict[str, object]:
        return {
            "id": self.sample_id,
            "title": self.title,
            "description": self.description,
            "image_path": self.image_uri,
            "overlay_path": self.overlay_uri,
            "cam_paths": self.cam_paths,
        }


class SampleStore:
    def __init__(self, settings: DemoSettings) -> None:
        self.settings = settings
        self._items: Dict[str, DemoSample] = {}
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        samples_dir = self.settings.samples_dir
        if not samples_dir.exists():
            self._items = {}
            return

        supported = {".png", ".jpg", ".jpeg"}
        samples: Dict[str, DemoSample] = {}

        for path in sorted(samples_dir.iterdir()):
            if path.suffix.lower() not in supported or not path.is_file():
                continue

            sample_id = path.stem
            title = sample_id.replace("_", " ").replace("-", " ")
            title = title.strip() or sample_id
            title = title.title()
            description = f"預設樣本：{sample_id}"

            relative_uri = f"/demo-assets/{self.settings.samples_subdir}/{path.name}"
            overlay_uri: Optional[str] = None
            cam_paths: Dict[str, str] = {}

            for ext in supported:
                candidate = path.with_name(f"{sample_id}_overlay{ext}")
                if candidate.exists():
                    overlay_uri = f"/demo-assets/{self.settings.samples_subdir}/{candidate.name}"
                    break

            prefix = f"{sample_id}_cam_"
            for cam_file in samples_dir.glob(f"{prefix}*"):
                if not cam_file.is_file() or cam_file.suffix.lower() not in supported:
                    continue
                fdi = cam_file.stem[len(prefix) :]
                if not fdi:
                    continue
                cam_paths[fdi] = f"/demo-assets/{self.settings.samples_subdir}/{cam_file.name}"

            samples[sample_id] = DemoSample(
                sample_id=sample_id,
                title=title,
                description=description,
                image_uri=relative_uri,
                image_path=path,
                overlay_uri=overlay_uri,
                cam_paths=cam_paths,
            )

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
