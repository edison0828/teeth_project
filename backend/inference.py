"""整合 YOLO 牙齒偵測與 Cross-Attention 齲齒病灶分類的推論服務。"""

from __future__ import annotations

import json
import re
import threading

from collections import Counter, OrderedDict

from dataclasses import dataclass

from datetime import datetime

from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

from uuid import uuid4



import cv2

import numpy as np

import torch

import torch.nn.functional as F
from torchvision import transforms



from .inference_models import build_cross_attn_fdi


def normalize_fdi_label(raw: Any) -> str:
    """Convert various FDI-style labels (e.g. ``FDI_11``) into pure digits."""

    if raw is None:
        return ""
    match = re.findall(r"\d+", str(raw))
    return match[0] if match else str(raw)


def fdi_to_basic_group(fdi_code: str) -> str:
    """Group FDI codes into incisor/premolar/molar buckets."""

    try:
        number = int(fdi_code)
    except (TypeError, ValueError):
        return "incisor"

    if number == 91:
        return "incisor"

    last_digit = number % 10
    if last_digit in (1, 2, 3):
        return "incisor"
    if last_digit in (4, 5):
        return "premolar"
    return "molar"





@dataclass(slots=True)

class FindingResult:

    tooth_label: str

    bbox_xyxy: List[float]

    detector_conf: float

    classifier_prob: float

    severity: str



    @property

    def bbox_xywh(self) -> List[float]:

        x1, y1, x2, y2 = self.bbox_xyxy

        return [x1, y1, x2 - x1, y2 - y1]





ModelType = Literal["cross_attn", "yolo_caries"]


@dataclass(frozen=True, slots=True)

class ModelDefinition:

    model_id: str

    model_type: ModelType

    detector_path: Path

    classifier_path: Path

    detector_threshold: float

    classification_threshold: float

    max_teeth: int

    updated_at: datetime



    @property

    def cache_key(self) -> str:

        timestamp = int(self.updated_at.timestamp())

        return (

            f"{self.model_id}:{self.model_type}:{self.detector_path}:{self.classifier_path}:"

            f"{self.detector_threshold:.4f}:{self.classification_threshold:.4f}:{self.max_teeth}:{timestamp}"

        )





class InferenceError(RuntimeError):

    """Raised when the inference engine fails to process an image."""





class BaseInference:

    """Common helpers shared by different inference pipelines."""

    def __init__(self) -> None:
        self._root = Path(__file__).resolve().parent.parent
        self._uploads_root = self._root / "uploaded_images"
        self._transform: Optional[transforms.Compose] = None
        self._fdi_to_index: Dict[str, int] = {}

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (self._root / path).resolve()

    def _compute_gradcam(
        self,
        sample: torch.Tensor,
        fdi_idx: torch.Tensor,
        original_patch: np.ndarray,
    ) -> Dict[str, Any]:
        assert self._classifier is not None and self._device is not None

        if sample.device != self._device:
            sample = sample.to(self._device)

        if fdi_idx.device != self._device:
            fdi_idx = fdi_idx.to(self._device)

        self._classifier.zero_grad(set_to_none=True)

        with torch.enable_grad():
            sample = sample.clone().detach().requires_grad_(True)
            logits, feat, _ = self._classifier(
                sample,
                fdi_idx,
                return_feat_for_cam=True,
                return_aux=True,
            )
            probs = F.softmax(logits, dim=1)
            target = probs[:, 1].sum()
            grads = torch.autograd.grad(
                target,
                feat,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            if grads is None:
                raise InferenceError("Unable to compute Grad-CAM gradients")

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * feat).sum(dim=1, keepdim=True))

        cam = F.interpolate(
            cam,
            size=(original_patch.shape[0], original_patch.shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        heatmap = (cam * 255).astype(np.uint8)

        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(colored, 0.45, original_patch, 0.55, 0)

        self._classifier.zero_grad(set_to_none=True)

        return {

            "probability": float(probs[0, 1].item()),

            "heatmap": heatmap,

            "colored": colored,

            "overlay": overlay,

        }

    def _ensure_output_dir(self, analysis_id: str) -> Path:
        base_dir = (self._uploads_root / "analysis").resolve()
        base_dir.mkdir(parents=True, exist_ok=True)

        candidate = (base_dir / analysis_id).resolve()
        if base_dir not in candidate.parents and candidate != base_dir:
            raise ValueError("Invalid analysis identifier")

        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _to_public_uri(self, path: Optional[Path | str]) -> Optional[str]:
        if path is None:
            return None

        path_str = str(path)
        if path_str.startswith(("http://", "https://")):
            return path_str
        if path_str.startswith("/"):
            return path_str

        candidate = Path(path_str)
        try:
            relative = candidate.resolve().relative_to(self._uploads_root.resolve())
        except Exception:
            return candidate.as_posix()

        return f"/uploaded_images/{relative.as_posix()}"

    @staticmethod
    def _severity_from_probability(probability: float) -> str:
        if probability >= 0.85:
            return "severe"
        if probability >= 0.6:
            return "moderate"
        return "mild"

    def _prepare_batch(self, patches: Iterable[np.ndarray], _: Iterable[int]) -> torch.Tensor:
        if self._transform is None:
            raise InferenceError("分類模型尚未初始化，無法進行推論")

        tensors: List[torch.Tensor] = []
        for patch in patches:
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb_patch)
            tensors.append(tensor)

        if not tensors:
            raise InferenceError("缺少可用的牙齒影像區塊")

        return torch.stack(tensors, dim=0)




    def analyze(
        self,
        image_path: Path,
        *,
        analysis_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_models()
        assert self._detector is not None and self._classifier is not None and self._device is not None

        if not image_path.exists():
            raise InferenceError(f"找不到影像檔案：{image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise InferenceError(f"無法讀取影像：{image_path}")

        height, width = image.shape[:2]
        overlay_image = image.copy()
        overlay_uri: Optional[str] = None

        output_dir: Optional[Path] = None
        if analysis_id:
            try:
                output_dir = self._ensure_output_dir(analysis_id)
            except Exception as exc:  # pragma: no cover - filesystem guard
                raise InferenceError(f"無法建立分析輸出資料夾：{exc}") from exc

        original_image_uri = self._to_public_uri(image_path)

        def build_response(
            findings: List[Dict[str, Any]],
            detected_conditions: List[Dict[str, Any]],
            summary: str,
            aggregate: float,
        ) -> Dict[str, Any]:
            return {
                "findings": findings,
                "detected_conditions": detected_conditions,
                "overall_assessment": summary,
                "aggregate_confidence": aggregate,
                "visualizations": {
                    "image_uri": original_image_uri,
                    "overlay_uri": overlay_uri,
                    "image_size": [width, height],
                },
            }

        inference = self._detector(
            str(image_path),
            conf=self.conf_threshold,
            verbose=False,
            device=str(self._device),
        )
        if not inference:
            return build_response([], [], "未偵測到牙齒。", 0.0)

        result = inference[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return build_response([], [], "未偵測到牙齒。", 0.0)

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        classes = boxes.cls.cpu().numpy().astype(int)

        masks_full: Optional[np.ndarray] = None
        kernel: Optional[np.ndarray] = None
        masks_attr = getattr(result, "masks", None)
        if masks_attr is not None:
            mask_data = getattr(masks_attr, "data", None)
            if mask_data is not None and len(mask_data) > 0:
                mask_np = mask_data.cpu().numpy()
                masks_full = cv2.resize(
                    mask_np.transpose(1, 2, 0),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).transpose(2, 0, 1)
                kernel = self._build_kernel()

        patches: List[np.ndarray] = []
        original_patches: List[np.ndarray] = []
        fdi_indices: List[int] = []
        meta: List[Dict[str, Any]] = []

        for idx, (bbox, det_conf, cls_idx) in enumerate(zip(xyxy, confs, classes)):
            if idx >= self.max_teeth:
                break

            x1, y1, x2, y2 = bbox.tolist()
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            fdi_label = (
                str(self._detector.names[cls_idx])
                if isinstance(self._detector.names, (dict, list))
                else str(cls_idx)
            )
            normalized_label = normalize_fdi_label(fdi_label)
            if normalized_label not in self._fdi_to_index:
                continue

            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            mask_for_overlay: Optional[np.ndarray] = None
            if (
                masks_full is not None
                and kernel is not None
                and idx < masks_full.shape[0]
            ):
                raw_mask = (masks_full[idx][y1:y2, x1:x2] > 0).astype(np.uint8) * 255
                if raw_mask.size > 0:
                    mask_for_overlay = self._process_mask(raw_mask, kernel)
                    if self._roi_style == "mask":
                        crop[mask_for_overlay == 0] = 0

            original_patches.append(crop.copy())
            resized = cv2.resize(
                crop,
                (self._patch_size, self._patch_size),
                interpolation=cv2.INTER_AREA,
            )
            patches.append(resized)
            fdi_indices.append(self._fdi_to_index[normalized_label])
            meta.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "detector_conf": float(det_conf),
                    "fdi_label": normalized_label,
                    "raw_label": fdi_label,
                    "detector_index": idx,
                }
            )

        if not patches:
            return build_response([], [], "未偵測到有效牙齒框。", 0.0)

        batch = self._prepare_batch(patches, fdi_indices).to(self._device)
        fdi_tensor = torch.tensor(fdi_indices, device=self._device, dtype=torch.long)

        with torch.no_grad():
            logits = self._classifier(batch, fdi_tensor)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

        severity_colors = {
            "mild": (0, 204, 255),
            "moderate": (0, 160, 255),
            "severe": (0, 102, 255),
        }

        serialized_findings: List[Dict[str, Any]] = []
        thresholds = [self._select_threshold(idx) for idx in fdi_indices]

        for idx, (info, prob, thr) in enumerate(zip(meta, probs, thresholds)):
            score = float(prob)
            if score < thr:
                continue
            severity = self._severity_from_probability(score)
            finding_id = f"FND-{uuid4().hex[:8].upper()}"

            x1, y1, x2, y2 = info["bbox"]
            x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
            box_w = max(x2_i - x1_i, 1)
            box_h = max(y2_i - y1_i, 1)
            bbox_xywh = [float(x1_i), float(y1_i), float(box_w), float(box_h)]
            bbox_normalized = [
                bbox_xywh[0] / width,
                bbox_xywh[1] / height,
                bbox_xywh[2] / width,
                bbox_xywh[3] / height,
            ]
            centroid = [
                (bbox_xywh[0] + bbox_xywh[2] / 2) / width,
                (bbox_xywh[1] + bbox_xywh[3] / 2) / height,
            ]

            color = severity_colors.get(severity, (0, 215, 255))
            cv2.rectangle(overlay_image, (x1_i, y1_i), (x2_i, y2_i), color, 2)
            label = f"{info['fdi_label']} p={score:.2f} thr={thr:.2f}"
            cv2.putText(
                overlay_image,
                label,
                (x1_i, max(y1_i - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

            if masks_full is not None and kernel is not None:
                det_index = info.get("detector_index")
                if isinstance(det_index, int) and 0 <= det_index < masks_full.shape[0]:
                    raw_mask = (masks_full[det_index][y1_i:y2_i, x1_i:x2_i] > 0).astype(np.uint8) * 255
                    if raw_mask.size > 0:
                        processed = self._process_mask(raw_mask, kernel)
                        overlay_patch = overlay_image[y1_i:y2_i, x1_i:x2_i]
                        overlay_image[y1_i:y2_i, x1_i:x2_i] = self._overlay_mask(
                            overlay_patch,
                            processed,
                            color=(0, 0, 255),
                            alpha=0.35,
                        )

            assets: Dict[str, Optional[str]] = {}
            if output_dir is not None:
                original_patch = original_patches[idx]
                crop_path = output_dir / f"{finding_id}_crop.png"
                cv2.imwrite(str(crop_path), original_patch)
                assets["crop"] = self._to_public_uri(crop_path)
                try:
                    gradcam = self._compute_gradcam(
                        batch[idx : idx + 1],
                        fdi_tensor[idx : idx + 1],
                        original_patch,
                    )
                except Exception:
                    gradcam = None
                if gradcam is not None:
                    heatmap_path = output_dir / f"{finding_id}_heatmap.png"
                    overlay_patch_path = output_dir / f"{finding_id}_gradcam.png"
                    cv2.imwrite(str(heatmap_path), gradcam["colored"])
                    cv2.imwrite(str(overlay_patch_path), gradcam["overlay"])
                    assets["heatmap"] = self._to_public_uri(heatmap_path)
                    assets["gradcam"] = self._to_public_uri(overlay_patch_path)

            assets = {key: value for key, value in assets.items() if value}

            serialized_findings.append(
                {
                    "finding_id": finding_id,
                    "type": "caries",
                    "tooth_label": f"FDI-{info['fdi_label']}",
                    "region": {
                        "bbox": [float(x1_i), float(y1_i), float(box_w), float(box_h)],
                        "mask_uri": None,
                    },
                    "severity": severity,
                    "confidence": round(score, 4),
                    "model_key": "cross_cam_caries",
                    "model_version": "camAlignA",
                    "extra": {
                        "detector_confidence": info["detector_conf"],
                        "classifier_probability": score,
                        "threshold_used": thr,
                        "raw_detector_label": info.get("raw_label"),
                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "bbox_normalized": bbox_normalized,
                        "centroid": centroid,
                        "assets": assets,
                        "color_bgr": list(color),
                        "image_id": image_id,
                    },
                }
            )

        if output_dir is not None:
            overlay_path = output_dir / "analysis_overlay.png"
            cv2.imwrite(str(overlay_path), overlay_image)
            overlay_uri = self._to_public_uri(overlay_path)

        if not serialized_findings:
            aggregate = float(probs.mean()) if len(probs) else 0.0
            return build_response([], [], "未檢出疑似齲齒病灶。", aggregate)

        for record in serialized_findings:
            extras = record.setdefault("extra", {})
            extras["analysis_overlay_uri"] = overlay_uri
            extras["image_size"] = {"width": width, "height": height}
            extras["image_uri"] = original_image_uri
            assets = extras.get("assets")
            if isinstance(assets, dict) and overlay_uri:
                assets.setdefault("overlay", overlay_uri)

        aggregate_conf = float(np.mean([record["confidence"] for record in serialized_findings]))
        severity_counter = Counter(record["severity"] for record in serialized_findings)
        total = sum(severity_counter.values())
        severity_breakdown = [
            {"level": level, "percentage": count / total}
            for level, count in severity_counter.items()
        ]

        detected_conditions = [
            {
                "label": "Caries",
                "count": len(serialized_findings),
                "severity_breakdown": severity_breakdown,
            }
        ]

        summary = f"偵測到 {len(serialized_findings)} 顆疑似齲齒 (平均信心 {aggregate_conf:.2f})"

        return build_response(serialized_findings, detected_conditions, summary, aggregate_conf)



class CrossCamInference(BaseInference):

    """Cross-attention classifier with YOLO tooth detector."""

    _MODEL_KWARGS = {"fdi_dim", "attn_dim", "heads", "num_queries", "use_film", "use_se"}

    def __init__(
        self,
        detector_path: Path,
        classifier_path: Path,
        *,
        conf_threshold: float = 0.25,
        caries_threshold: float = 0.5,
        max_teeth: int = 64,
    ) -> None:
        super().__init__()
        self.detector_path = self._resolve_path(detector_path)
        self.classifier_path = self._resolve_path(classifier_path)
        self.conf_threshold = conf_threshold
        self.caries_threshold = caries_threshold
        self.max_teeth = max_teeth

        self._lock = threading.Lock()
        self._device: Optional[torch.device] = None
        self._detector = None
        self._classifier = None
        self._patch_size = 224
        self._roi_style = "bbox"
        self._dilate_kernel = 7
        self._dilate_iter = 6
        self._smooth_blur = True
        self._smooth_iters = 3
        self._apply_gaussian = True
        self._threshold_mode = "fixed"
        self._layered_thresholds: Optional[Dict[str, Any]] = None
        self._idx_to_fdi: Dict[int, str] = {}

        self._update_transform()

    def _update_transform(self) -> None:
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def _load_layered_thresholds(self, payload: Any) -> Optional[Dict[str, Any]]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str) and payload:
            path = Path(payload)
            if not path.is_absolute():
                path = self.classifier_path.parent / path
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    if isinstance(data, dict):
                        return data
                except Exception:  # pragma: no cover - best effort parse
                    return None
        return None

    def _apply_metadata_config(self, metadata: Dict[str, Any]) -> None:
        if not metadata:
            self._update_transform()
            return

        sources: List[Dict[str, Any]] = [metadata]
        for key in ("inference", "analysis", "analysis_config", "config"):
            nested = metadata.get(key)
            if isinstance(nested, dict):
                sources.append(nested)

        for source in sources:
            roi_style = source.get("roi_style") or source.get("roiStyle")
            if isinstance(roi_style, str):
                self._roi_style = roi_style.strip().lower() or self._roi_style

            patch_size = source.get("patch_size") or source.get("image_size")
            if patch_size is not None:
                self._patch_size = max(8, self._coerce_int(patch_size, self._patch_size))

            dilate_kernel = source.get("dilate_kernel") or source.get("mask_kernel")
            if dilate_kernel is not None:
                self._dilate_kernel = max(1, self._coerce_int(dilate_kernel, self._dilate_kernel))

            dilate_iter = source.get("dilate_iter") or source.get("dilate_iterations")
            if dilate_iter is not None:
                self._dilate_iter = max(0, self._coerce_int(dilate_iter, self._dilate_iter))

            smooth_blur = source.get("smooth_blur")
            if smooth_blur is not None:
                self._smooth_blur = self._coerce_bool(smooth_blur, self._smooth_blur)

            smooth_iters = source.get("smooth_iters") or source.get("smooth_iterations")
            if smooth_iters is not None:
                self._smooth_iters = max(0, self._coerce_int(smooth_iters, self._smooth_iters))

            gaussian = source.get("apply_gaussian") or source.get("gaussian")
            if gaussian is not None:
                self._apply_gaussian = self._coerce_bool(gaussian, self._apply_gaussian)

            fallback_thr = source.get("threshold") or source.get("fallback_threshold")
            if fallback_thr is not None:
                self.caries_threshold = self._coerce_float(fallback_thr, self.caries_threshold)

            mode = source.get("thr_mode") or source.get("threshold_mode")
            if isinstance(mode, str):
                self._threshold_mode = mode.strip().lower() or self._threshold_mode

            layered = (
                source.get("layered_thresholds")
                or source.get("thresholds")
                or source.get("layered_thresholds_path")
                or source.get("threshold_config")
            )
            if layered:
                loaded = self._load_layered_thresholds(layered)
                if loaded:
                    self._layered_thresholds = loaded

        self._update_transform()

    def _build_kernel(self) -> np.ndarray:
        size = max(1, int(self._dilate_kernel))
        if size % 2 == 0:
            size += 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _process_mask(self, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        processed = mask.copy()
        if self._dilate_iter > 0:
            processed = cv2.dilate(processed, kernel, iterations=self._dilate_iter)
        if self._smooth_blur and self._smooth_iters > 0:
            processed = cv2.morphologyEx(
                processed,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=self._smooth_iters,
            )
        if self._apply_gaussian:
            processed = cv2.GaussianBlur(processed, (5, 5), 2)
        _, processed = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)
        return processed

    @staticmethod
    def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
        overlay = image.copy()
        idx = mask > 0
        overlay[idx] = (
            (1 - alpha) * overlay[idx] + alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)
        return overlay

    def _pick_threshold_from_config(self, fdi_idx: int) -> float:
        if not self._layered_thresholds or fdi_idx not in self._idx_to_fdi:
            return self.caries_threshold

        config = self._layered_thresholds
        fdi_code = self._idx_to_fdi.get(fdi_idx)
        if not fdi_code:
            return self.caries_threshold

        per_fdi = config.get("per_fdi")
        if isinstance(per_fdi, dict) and fdi_code in per_fdi:
            return self._coerce_float(per_fdi[fdi_code], self.caries_threshold)

        grouping = "basic"
        meta = config.get("meta")
        if isinstance(meta, dict):
            grouping = str(meta.get("grouping", grouping))

        if grouping == "basic":
            group_name = fdi_to_basic_group(fdi_code)
        else:
            group_name = None

        groups = config.get("groups")
        if group_name and isinstance(groups, dict) and group_name in groups:
            return self._coerce_float(groups[group_name], self.caries_threshold)

        global_thr = config.get("global")
        if global_thr is not None:
            return self._coerce_float(global_thr, self.caries_threshold)

        return self.caries_threshold

    def _select_threshold(self, fdi_idx: int) -> float:
        if self._threshold_mode == "layered":
            return self._pick_threshold_from_config(fdi_idx)
        return self.caries_threshold

    def _ensure_models(self) -> None:
        if self._detector is not None and self._classifier is not None:
            return

        with self._lock:
            if self._device is None:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if self._detector is None:
                try:
                    from ultralytics import YOLO
                except ImportError as exc:  # pragma: no cover - runtime dependency
                    raise InferenceError("缺少 ultralytics 套件，無法啟動牙齒偵測模型") from exc

                if not self.detector_path.exists():
                    raise InferenceError(f"偵測模型不存在: {self.detector_path}")

                self._detector = YOLO(str(self.detector_path))
                if hasattr(self._detector, "fuse"):
                    try:
                        self._detector.fuse()
                    except Exception:  # pragma: no cover - best effort fuse
                        pass

            if self._classifier is None:
                if not self.classifier_path.exists():
                    raise InferenceError(f"齲齒分類模型不存在: {self.classifier_path}")

                checkpoint = torch.load(
                    str(self.classifier_path),
                    map_location="cpu",
                )
                state_dict = self._extract_state_dict(checkpoint)
                state_dict = self._normalise_state_dict_keys(state_dict)
                metadata = self._extract_metadata(checkpoint)
                self._apply_metadata_config(metadata)
                model_kwargs = {
                    key: metadata[key]
                    for key in self._MODEL_KWARGS
                    if key in metadata
                }
                num_fdi = self._infer_num_fdi(state_dict)

                classifier = build_cross_attn_fdi(num_fdi=num_fdi, **model_kwargs)
                try:
                    classifier.load_state_dict(state_dict, strict=True)
                except RuntimeError as exc:
                    incompatibilities = classifier.load_state_dict(state_dict, strict=False)
                    missing = sorted(incompatibilities.missing_keys)
                    unexpected = sorted(incompatibilities.unexpected_keys)
                    details = []
                    if missing:
                        details.append(f"缺少權重: {', '.join(missing)}")
                    if unexpected:
                        details.append(f"多餘權重: {', '.join(unexpected)}")
                    message = "分類模型權重載入失敗"
                    if details:
                        message = f"{message}（{'；'.join(details)}）"
                    raise InferenceError(message) from exc
                classifier.to(self._device)
                classifier.eval()
                self._classifier = classifier

                mapping = self._extract_fdi_mapping(checkpoint)
                if not mapping:
                    mapping = self._build_mapping_from_detector(num_fdi)
                mapping = self._normalise_mapping(mapping, num_fdi)
                if not mapping:
                    mapping = self._sequential_mapping(num_fdi)
                self._idx_to_fdi = {idx: label for label, idx in mapping.items()}
                self._fdi_to_index = self._expand_with_synonyms(mapping)

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state", "model", "net"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            if all(isinstance(k, str) for k in checkpoint.keys()):
                return checkpoint  # already a state dict
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint.state_dict()
        if isinstance(checkpoint, (list, tuple)):
            raise InferenceError("不支援的分類模型格式：列表/元組")
        if not isinstance(checkpoint, dict):
            raise InferenceError("無法讀取分類模型權重")
        return checkpoint

    @staticmethod
    def _normalise_state_dict_keys(
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not state_dict:
            return state_dict

        items = []
        seen = set()
        for key, value in state_dict.items():
            new_key = key
            while new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key in seen:
                raise InferenceError(f"分類模型包含重複權重鍵：{new_key}")
            seen.add(new_key)
            items.append((new_key, value))

        if isinstance(state_dict, OrderedDict):
            return OrderedDict(items)
        return dict(items)

    @staticmethod
    def _extract_metadata(checkpoint: Any) -> Dict[str, Any]:
        if not isinstance(checkpoint, dict):
            return {}
        metadata: Dict[str, Any] = {}
        meta = checkpoint.get("metadata")
        if isinstance(meta, dict):
            metadata.update(meta)
        config = checkpoint.get("config")
        if isinstance(config, dict):
            metadata.update(config)
        return metadata

    @staticmethod
    def _infer_num_fdi(state_dict: Dict[str, torch.Tensor]) -> int:
        for key in ("fdi_emb.weight", "module.fdi_emb.weight"):
            if key in state_dict:
                weight = state_dict[key]
                if hasattr(weight, "shape"):
                    return int(weight.shape[0])
        for key, value in state_dict.items():
            if key.endswith("fdi_emb.weight") and hasattr(value, "shape"):
                return int(value.shape[0])
        raise InferenceError("無法從分類模型檔案推斷 FDI 種類數")

    def _extract_fdi_mapping(self, checkpoint: Any) -> Dict[str, int]:
        if not isinstance(checkpoint, dict):
            return {}

        candidates: List[Any] = []
        for key in ("fdi_to_idx", "fdi_to_index", "fdi_mapping"):
            if key in checkpoint:
                candidates.append(checkpoint[key])

        metadata = self._extract_metadata(checkpoint)
        for key in ("fdi_to_idx", "fdi_to_index", "fdi_mapping", "idx_to_fdi", "fdi_labels"):
            if key in metadata:
                candidates.append(metadata[key])

        for raw in candidates:
            mapping = self._normalise_raw_mapping(raw)
            if mapping:
                return mapping

        return {}

    @staticmethod
    def _normalise_raw_mapping(raw: Any) -> Dict[str, int]:
        if isinstance(raw, dict):
            result: Dict[str, int] = {}
            for key, value in raw.items():
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                label = normalize_fdi_label(key)
                if not label:
                    continue
                result[label] = idx
            return result
        if isinstance(raw, (list, tuple)):
            mapping: Dict[str, int] = {}
            for idx, item in enumerate(raw):
                label = normalize_fdi_label(item)
                if label and label not in mapping:
                    mapping[label] = idx
            return mapping
        return {}

    def _build_mapping_from_detector(self, num_fdi: int) -> Dict[str, int]:
        names_source = getattr(self._detector, "names", None)
        labels: List[str] = []
        if isinstance(names_source, dict):
            labels = [str(name) for _, name in sorted(names_source.items(), key=lambda item: item[0])]
        elif isinstance(names_source, list):
            labels = [str(name) for name in names_source]

        labels = [normalize_fdi_label(label.strip()) for label in labels if label]
        seen: Dict[str, None] = {}
        unique_labels: List[str] = []
        for label in labels:
            if label and label not in seen:
                seen[label] = None
                unique_labels.append(label)

        if len(unique_labels) < num_fdi:
            unique_labels = self._default_fdi_codes(num_fdi)

        mapping: Dict[str, int] = {}
        for idx, label in enumerate(unique_labels[:num_fdi]):
            mapping[label] = idx
        return mapping

    @staticmethod
    def _normalise_mapping(mapping: Dict[str, int], num_fdi: int) -> Dict[str, int]:
        canonical: Dict[int, str] = {}
        for key, value in mapping.items():
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if not 0 <= idx < num_fdi:
                continue
            label = normalize_fdi_label(key)
            if not label:
                continue
            canonical.setdefault(idx, label)

        if len(canonical) != num_fdi:
            return {}

        ordered = sorted(canonical.items(), key=lambda item: item[0])
        return {label: idx_value for idx_value, label in ordered}

    @staticmethod
    def _sequential_mapping(num_fdi: int) -> Dict[str, int]:
        return {str(code): idx for idx, code in enumerate(CrossCamInference._default_fdi_codes(num_fdi))}

    @staticmethod
    def _default_fdi_codes(num_fdi: int) -> List[str]:
        permanent = [f"{quadrant}{tooth}" for quadrant in (1, 2, 3, 4) for tooth in range(1, 9)]
        primary = [f"{quadrant}{tooth}" for quadrant in (5, 6, 7, 8) for tooth in range(1, 6)]
        codes = permanent + primary
        if num_fdi > len(codes):
            extra = [str(idx) for idx in range(num_fdi - len(codes))]
            codes.extend(extra)
        return codes[:num_fdi]

    @staticmethod
    def _expand_with_synonyms(mapping: Dict[str, int]) -> Dict[str, int]:
        expanded: Dict[str, int] = {}
        for label, idx in mapping.items():
            normalized = normalize_fdi_label(label)
            candidates = {
                label,
                normalized,
                f"FDI-{normalized}" if normalized else None,
                f"FDI_{normalized}" if normalized else None,
            }
            for candidate in candidates:
                if candidate:
                    expanded.setdefault(candidate, idx)
        return expanded



class YoloCariesInference(BaseInference):

    """Two-stage pipeline: YOLO teeth localisation + YOLO caries detection."""

    def __init__(
        self,
        detector_path: Path,
        caries_path: Path,
        *,
        conf_threshold: float = 0.25,
        caries_threshold: float = 0.35,
        max_teeth: int = 64,
        match_iou_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.detector_path = self._resolve_path(detector_path)
        self.caries_path = self._resolve_path(caries_path)
        self.conf_threshold = conf_threshold
        self.caries_threshold = caries_threshold
        self.max_teeth = max_teeth
        self.match_iou_threshold = match_iou_threshold

        self._lock = threading.Lock()
        self._device: Optional[torch.device] = None
        self._tooth_detector = None
        self._caries_detector = None

    def _ensure_models(self) -> None:
        if self._tooth_detector is not None and self._caries_detector is not None:
            return

        with self._lock:
            if self._device is None:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if self._tooth_detector is None:
                try:
                    from ultralytics import YOLO
                except ImportError as exc:  # pragma: no cover - runtime dependency
                    raise InferenceError("缺少 ultralytics 套件，無法啟動牙齒偵測模型") from exc

                if not self.detector_path.exists():
                    raise InferenceError(f"偵測模型不存在: {self.detector_path}")

                self._tooth_detector = YOLO(str(self.detector_path))
                if hasattr(self._tooth_detector, "fuse"):
                    try:
                        self._tooth_detector.fuse()
                    except Exception:  # pragma: no cover - best effort fuse
                        pass

            if self._caries_detector is None:
                try:
                    from ultralytics import YOLO
                except ImportError as exc:  # pragma: no cover - runtime dependency
                    raise InferenceError("缺少 ultralytics 套件，無法啟動齲齒偵測模型") from exc

                if not self.caries_path.exists():
                    raise InferenceError(f"齲齒偵測模型不存在: {self.caries_path}")

                self._caries_detector = YOLO(str(self.caries_path))
                if hasattr(self._caries_detector, "fuse"):
                    try:
                        self._caries_detector.fuse()
                    except Exception:  # pragma: no cover - best effort fuse
                        pass

    @staticmethod
    def _bbox_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom

    @staticmethod
    def _bbox_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _assign_tooth(
        self,
        caries_box: Tuple[float, float, float, float],
        teeth: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], float, str]:
        if not teeth:
            return None, 0.0, "none"

        best_tooth = None
        best_iou = 0.0
        for tooth in teeth:
            iou = self._bbox_iou(caries_box, tooth["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_tooth = tooth

        if best_tooth is not None and best_iou >= self.match_iou_threshold:
            return best_tooth, best_iou, "iou"

        cx, cy = self._bbox_center(caries_box)
        nearest = min(
            teeth,
            key=lambda tooth: (
                (self._bbox_center(tooth["bbox"])[0] - cx) ** 2
                + (self._bbox_center(tooth["bbox"])[1] - cy) ** 2
            ),
        )
        return nearest, best_iou, "centroid"

    def analyze(
        self,
        image_path: Path,
        *,
        analysis_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_models()
        assert self._tooth_detector is not None and self._caries_detector is not None and self._device is not None

        if not image_path.exists():
            raise InferenceError(f"找不到影像檔案：{image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise InferenceError(f"無法讀取影像：{image_path}")

        height, width = image.shape[:2]
        overlay_image = image.copy()
        overlay_uri: Optional[str] = None

        output_dir: Optional[Path] = None
        if analysis_id:
            try:
                output_dir = self._ensure_output_dir(analysis_id)
            except Exception as exc:  # pragma: no cover - filesystem guard
                raise InferenceError(f"無法建立分析輸出資料夾：{exc}") from exc

        original_image_uri = self._to_public_uri(image_path)

        def build_response(
            findings: List[Dict[str, Any]],
            detected_conditions: List[Dict[str, Any]],
            summary: str,
            aggregate: float,
        ) -> Dict[str, Any]:
            return {
                "findings": findings,
                "detected_conditions": detected_conditions,
                "overall_assessment": summary,
                "aggregate_confidence": aggregate,
                "visualizations": {
                    "image_uri": original_image_uri,
                    "overlay_uri": overlay_uri,
                    "image_size": [width, height],
                },
            }

        tooth_inference = self._tooth_detector(
            str(image_path),
            conf=self.conf_threshold,
            verbose=False,
            device=str(self._device),
        )
        teeth_meta: List[Dict[str, Any]] = []
        if tooth_inference:
            tooth_result = tooth_inference[0]
            boxes = tooth_result.boxes
            if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
                classes = boxes.cls.cpu().numpy().astype(int)
                names_source = getattr(self._tooth_detector, "names", None)
                for idx, (bbox, det_conf, cls_idx) in enumerate(zip(xyxy, confs, classes)):
                    if idx >= self.max_teeth:
                        break
                    x1, y1, x2, y2 = bbox.tolist()
                    x1_i, y1_i = max(0, int(round(x1))), max(0, int(round(y1)))
                    x2_i, y2_i = min(width, int(round(x2))), min(height, int(round(y2)))
                    if x2_i - x1_i < 8 or y2_i - y1_i < 8:
                        continue
                    if isinstance(names_source, dict):
                        label = str(names_source.get(cls_idx, cls_idx))
                    elif isinstance(names_source, list):
                        label = str(names_source[cls_idx]) if 0 <= cls_idx < len(names_source) else str(cls_idx)
                    else:
                        label = str(cls_idx)
                    teeth_meta.append(
                        {
                            "label": label,
                            "bbox": (float(x1_i), float(y1_i), float(x2_i), float(y2_i)),
                            "detector_conf": float(det_conf),
                        }
                    )
                    cv2.rectangle(overlay_image, (x1_i, y1_i), (x2_i, y2_i), (90, 180, 255), 1)
                    cv2.putText(
                        overlay_image,
                        f"{label}",
                        (x1_i, max(0, y1_i - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (90, 180, 255),
                        1,
                        cv2.LINE_AA,
                    )

        caries_inference = self._caries_detector(
            str(image_path),
            conf=self.caries_threshold,
            verbose=False,
            device=str(self._device),
        )

        if not caries_inference:
            return build_response([], [], "未檢出疑似齲齒病灶。", 0.0)

        result = caries_inference[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return build_response([], [], "未檢出疑似齲齒病灶。", 0.0)

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

        serialized_findings: List[Dict[str, Any]] = []
        confidences: List[float] = []

        for idx, (bbox, det_conf) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = bbox.tolist()
            x1_i, y1_i = max(0, int(round(x1))), max(0, int(round(y1)))
            x2_i, y2_i = min(width, int(round(x2))), min(height, int(round(y2)))
            if x2_i - x1_i < 4 or y2_i - y1_i < 4:
                continue

            match_tooth, match_iou, match_strategy = self._assign_tooth(
                (float(x1_i), float(y1_i), float(x2_i), float(y2_i)),
                teeth_meta,
            )
            tooth_label = None
            tooth_conf = None
            if match_tooth is not None:
                tooth_label = f"FDI-{match_tooth['label']}"
                tooth_conf = match_tooth.get("detector_conf")

            box_w, box_h = x2_i - x1_i, y2_i - y1_i
            bbox_normalized = [
                x1_i / width,
                y1_i / height,
                box_w / width,
                box_h / height,
            ]
            centroid = [
                (x1_i + box_w / 2.0) / width,
                (y1_i + box_h / 2.0) / height,
            ]

            score = float(det_conf)
            confidences.append(score)
            severity = self._severity_from_probability(score)

            finding_id = f"YOLO-{idx + 1:02d}-{uuid4().hex[:6]}"
            crop = image[y1_i:y2_i, x1_i:x2_i]
            assets: Dict[str, Optional[str]] = {}
            if output_dir is not None and crop.size > 0:
                crop_path = output_dir / f"{finding_id}_crop.png"
                cv2.imwrite(str(crop_path), crop)
                assets["crop"] = self._to_public_uri(crop_path)

            cv2.rectangle(overlay_image, (x1_i, y1_i), (x2_i, y2_i), (0, 68, 255), 2)
            cv2.putText(
                overlay_image,
                f"{score:.2f}",
                (x1_i, max(0, y1_i - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 68, 255),
                1,
                cv2.LINE_AA,
            )

            serialized_findings.append(
                {
                    "finding_id": finding_id,
                    "type": "caries",
                    "tooth_label": tooth_label,
                    "region": {
                        "bbox": [float(x1_i), float(y1_i), float(box_w), float(box_h)],
                        "mask_uri": None,
                    },
                    "severity": severity,
                    "confidence": round(score, 4),
                    "model_key": "yolo_caries",
                    "model_version": "yolo_caries_v1",
                    "extra": {
                        "detector_confidence": score,
                        "bbox_xyxy": [float(x1_i), float(y1_i), float(x2_i), float(y2_i)],
                        "bbox_normalized": bbox_normalized,
                        "centroid": centroid,
                        "assignment_iou": match_iou,
                        "assignment_strategy": match_strategy,
                        "matched_tooth_confidence": tooth_conf,
                        "image_id": image_id,
                        "assets": assets,
                    },
                }
            )

        if output_dir is not None:
            overlay_path = output_dir / "analysis_overlay.png"
            cv2.imwrite(str(overlay_path), overlay_image)
            overlay_uri = self._to_public_uri(overlay_path)

        if not serialized_findings:
            aggregate = float(np.mean(confidences)) if confidences else 0.0
            return build_response([], [], "未檢出疑似齲齒病灶。", aggregate)

        for record in serialized_findings:
            extras = record.setdefault("extra", {})
            extras["analysis_overlay_uri"] = overlay_uri
            extras["image_size"] = {"width": width, "height": height}
            extras["image_uri"] = original_image_uri
            assets = extras.get("assets")
            if isinstance(assets, dict) and overlay_uri:
                assets.setdefault("overlay", overlay_uri)

        aggregate_conf = float(np.mean([record["confidence"] for record in serialized_findings]))
        severity_counter = Counter(record["severity"] for record in serialized_findings)
        total = sum(severity_counter.values())
        severity_breakdown = [
            {"level": level, "percentage": count / total}
            for level, count in severity_counter.items()
        ]

        detected_conditions = [
            {
                "label": "Caries",
                "count": len(serialized_findings),
                "severity_breakdown": severity_breakdown,
            }
        ]

        summary = f"偵測到 {len(serialized_findings)} 顆疑似齲齒 (平均信心 {aggregate_conf:.2f})"

        return build_response(serialized_findings, detected_conditions, summary, aggregate_conf)



_engine_cache: Dict[str, BaseInference] = {}

_engine_lock = threading.Lock()





def get_inference_engine(definition: ModelDefinition) -> BaseInference:

    cache_key = definition.cache_key

    cached = _engine_cache.get(cache_key)

    if cached is not None:

        return cached

    with _engine_lock:

        cached = _engine_cache.get(cache_key)

        if cached is not None:

            return cached

        if definition.model_type == "cross_attn":

            engine = CrossCamInference(

                detector_path=definition.detector_path,

                classifier_path=definition.classifier_path,

                conf_threshold=definition.detector_threshold,

                caries_threshold=definition.classification_threshold,

                max_teeth=definition.max_teeth,

            )

        elif definition.model_type == "yolo_caries":

            engine = YoloCariesInference(

                detector_path=definition.detector_path,

                caries_path=definition.classifier_path,

                conf_threshold=definition.detector_threshold,

                caries_threshold=definition.classification_threshold,

                max_teeth=definition.max_teeth,

            )

        else:

            raise InferenceError(f"不支援的模型型態：{definition.model_type}")

        _engine_cache.clear()

        _engine_cache[cache_key] = engine

        return engine
