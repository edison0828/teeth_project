from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DemoSampleFinding(BaseModel):
    fdi: str
    prob_caries: float = Field(..., ge=0.0, le=1.0)
    thr_used: float
    pred: bool
    bbox: BoundingBox


class DemoSampleSummary(BaseModel):
    id: str
    title: str
    description: str
    image_path: str
    overlay_path: Optional[str] = None
    cam_paths: Dict[str, str] = Field(default_factory=dict)


class DemoInferenceFinding(DemoSampleFinding):
    orig_image: Optional[str] = None
    cam_path: Optional[str] = None
    roi_path: Optional[str] = None
    roi_path: Optional[str] = None


class DemoInferenceResponse(BaseModel):
    request_id: str
    overlay_url: str
    csv_url: str
    findings: List[DemoInferenceFinding]
    warnings: List[str] = []


class DemoSampleListResponse(BaseModel):
    items: List[DemoSampleSummary]


class DemoError(BaseModel):
    detail: str
