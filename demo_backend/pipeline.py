from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4
from threading import Lock
import shutil
import csv

import torch
from ultralytics import YOLO

from src.infer_cross_cam import (
    FDI_TO_IDX,
    IDX_TO_FDI,
    build_classifier,
    infer_one_image,
)

from .config import DemoSettings


@dataclass(slots=True)
class DemoInferenceArgs:
    """Arguments mirrored from ``infer_cross_cam.py`` for reuse."""

    # detector options
    yolo_conf: float = 0.25
    roi_style: str = "bbox"

    # classifier architecture
    use_se: bool = True
    use_film: bool = True
    num_queries: int = 4
    attn_dim: int = 256
    attn_heads: int = 8
    fdi_dim: int = 32
    clf_path: str = ""

    # preprocessing
    patch_size: int = 224
    dilate_kernel: int = 7
    dilate_iter: int = 6
    smooth_blur: bool = True
    smooth_iters: int = 3
    apply_gaussian: bool = True
    dump_rois: bool = False
    draw_normal: bool = True

    # thresholding
    thr_mode: str = "fixed"
    threshold: float = 0.5

    # runtime
    save_dir: str = ""


@dataclass(slots=True)
class DemoPrediction:
    request_id: str
    overlay_path: Path
    csv_path: Path
    findings: List[Dict[str, object]]
    warnings: List[str]

    def overlay_url(self, settings: DemoSettings) -> str:
        rel = self.overlay_path.relative_to(settings.output_dir)
        return f"/demo-outputs/{rel.as_posix()}"

    def csv_url(self, settings: DemoSettings) -> str:
        rel = self.csv_path.relative_to(settings.output_dir)
        return f"/demo-outputs/{rel.as_posix()}"


class DemoInferenceError(RuntimeError):
    """Raised when the pipeline fails to produce a valid prediction."""


class CrossAttentionDemoPipeline:
    def __init__(self, settings: DemoSettings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._yolo: Optional[YOLO] = None
        self._classifier: Optional[torch.nn.Module] = None
        self._thr_cfg: Optional[Dict[str, object]] = None
        self._base_args = DemoInferenceArgs(
            clf_path=str(settings.classifier_weights),
            draw_normal=True,
        )
        self._device = torch.device(
            "cuda"
            if settings.device.lower() == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

        if settings.layered_thresholds and settings.layered_thresholds.exists():
            with open(settings.layered_thresholds, "r", encoding="utf-8") as f:
                self._thr_cfg = json.load(f)
            self._base_args = replace(self._base_args, thr_mode="layered")

    # ------------------------------------------------------------------
    def ensure_loaded(self) -> None:
        with self._lock:
            if self._yolo is None:
                if not self.settings.yolo_weights.exists():
                    raise FileNotFoundError(
                        f"YOLO weights not found: {self.settings.yolo_weights}"
                    )
                self._yolo = YOLO(str(self.settings.yolo_weights))
                self._yolo.fuse()

            if self._classifier is None:
                if not self.settings.classifier_weights.exists():
                    raise FileNotFoundError(
                        f"Classifier weights not found: {self.settings.classifier_weights}"
                    )
                args = replace(self._base_args, clf_path=str(self.settings.classifier_weights))
                model = build_classifier(args, num_fdi=len(FDI_TO_IDX))
                model.to(self._device)
                model.eval()
                self._classifier = model

    # ------------------------------------------------------------------
    def unload(self) -> None:
        with self._lock:
            self._yolo = None
            self._classifier = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------
    def predict(self, image_path: Path, request_id: Optional[str] = None) -> DemoPrediction:
        self.ensure_loaded()
        assert self._yolo is not None and self._classifier is not None  # for type checkers

        req_id = request_id or uuid4().hex
        out_dir = self.settings.output_dir / req_id
        out_dir.mkdir(parents=True, exist_ok=True)

        args = replace(
            self._base_args,
            clf_path=str(self.settings.classifier_weights),
            save_dir=str(out_dir),
        )

        try:
            result = infer_one_image(
                str(image_path),
                self._yolo,
                self._classifier,
                self._device,
                args,
                FDI_TO_IDX,
                IDX_TO_FDI,
                thr_cfg=self._thr_cfg,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            shutil.rmtree(out_dir, ignore_errors=True)
            raise DemoInferenceError(str(exc)) from exc

        if result is None:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise DemoInferenceError("No teeth detected by YOLO detector")

        overlay_path_str, csv_path_str = result
        overlay_path = Path(overlay_path_str).resolve()
        csv_path = Path(csv_path_str).resolve()

        if not overlay_path.exists() or not csv_path.exists():
            raise DemoInferenceError("Inference outputs are missing")

        findings: List[Dict[str, object]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                findings.append(
                    {
                        "orig_image": row.get("orig_image"),
                        "fdi": row.get("fdi"),
                        "prob_caries": float(row.get("prob_caries", 0.0)),
                        "thr_used": float(row.get("thr_used", 0.0)),
                        "pred": bool(int(row.get("pred", 0))),
                        "bbox": {
                            "x1": int(float(row.get("x1", 0))),
                            "y1": int(float(row.get("y1", 0))),
                            "x2": int(float(row.get("x2", 0))),
                            "y2": int(float(row.get("y2", 0))),
                        },
                    }
                )

        warnings: List[str] = []
        if not findings:
            warnings.append("No tooth findings were generated from the classifier output.")

        return DemoPrediction(
            request_id=req_id,
            overlay_path=overlay_path,
            csv_path=csv_path,
            findings=findings,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    def warmup(self, sample_image: Optional[Path] = None) -> None:
        try:
            self.ensure_loaded()
        except FileNotFoundError:
            return

        if sample_image and sample_image.exists():
            try:
                self.predict(sample_image, request_id="warmup")
            except DemoInferenceError:
                pass
            finally:
                warmup_dir = self.settings.output_dir / "warmup"
                if warmup_dir.exists():
                    shutil.rmtree(warmup_dir, ignore_errors=True)
