from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional


@dataclass(slots=True)
class DemoSettings:
    """Runtime configuration for the demo inference API."""

    yolo_weights: Path = Path(os.getenv("DEMO_YOLO_WEIGHTS", "models/fdi_seg.pt"))
    classifier_weights: Path = Path(
        os.getenv("DEMO_CLASSIFIER_WEIGHTS", "models/cross_attn_fdi_camAlignA.pth")
    )
    layered_thresholds: Optional[Path] = (
        Path(thr_path) if (thr_path := os.getenv("DEMO_LAYERED_THRESHOLDS")) else None
    )
    output_dir: Path = Path(os.getenv("DEMO_OUTPUT_DIR", "demo_backend/outputs"))
    static_dir: Path = Path(os.getenv("DEMO_STATIC_DIR", "demo_backend/static"))
    samples_subdir: str = os.getenv("DEMO_SAMPLES_SUBDIR", "samples")
    device_preference: str = os.getenv("DEMO_DEVICE", "cuda")
    autoload_model: bool = os.getenv("DEMO_AUTOLOAD", "false").lower() in {"1", "true", "yes"}

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        (self.static_dir / self.samples_subdir).mkdir(parents=True, exist_ok=True)

    @property
    def samples_dir(self) -> Path:
        return self.static_dir / self.samples_subdir

    @property
    def device(self) -> str:
        return self.device_preference
