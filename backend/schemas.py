"""Pydantic models describing the Oral X-Ray analytics domain."""
from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, EmailStr, Field, ConfigDict


class QuickAction(BaseModel):
    """Represents a recommended quick action on the dashboard."""

    id: str
    label: str
    description: Optional[str] = None
    action: str = Field(
        ..., description="API endpoint or URL the action should direct to."
    )


class SystemStatus(BaseModel):
    pending_images: int = Field(..., description="Number of images waiting for review")
    new_reports: int = Field(..., description="New analysis reports awaiting review")
    models_active: int = Field(..., description="Enabled AI models")
    active_model_name: Optional[str] = None
    last_synced: datetime


class StatisticsOverview(BaseModel):
    weekly_volume: int
    week_over_week_change: float = Field(..., description="Percentage change vs last week")
    detection_rate: float = Field(..., ge=0, le=1)
    average_processing_time: float = Field(..., description="Minutes")
    uptime_percentage: float = Field(..., ge=0, le=1)


class ConditionSummary(BaseModel):
    label: str
    count: int
    severity_breakdown: List["SeveritySlice"]


class SeveritySlice(BaseModel):
    level: str
    percentage: float = Field(..., ge=0, le=1)


class PatientSummary(BaseModel):
    id: str
    name: str
    last_visit: Optional[date]
    most_recent_study: Optional[str] = Field(
        None, description="Image type or analysis tied to the latest visit"
    )


class ImageQueueItem(BaseModel):
    id: str
    patient_id: str
    patient_name: str
    image_type: str
    submitted_at: datetime
    status: str


class DashboardOverview(BaseModel):
    quick_actions: List[QuickAction]
    system_status: SystemStatus
    statistics: StatisticsOverview
    detected_conditions: List[ConditionSummary]
    recent_patients: List[PatientSummary]
    pending_images: List[ImageQueueItem]


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None


class ChangePassword(BaseModel):
    current_password: str = Field(..., min_length=8, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


class UserPublic(UserBase):
    id: str
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: Literal['bearer'] = 'bearer'
    expires_in: int


class PatientBase(BaseModel):
    name: str
    dob: date
    gender: str
    contact: Optional[str] = None
    medical_history: Optional[str] = None


class Patient(PatientBase):
    id: str
    last_visit: Optional[date] = None
    notes: Optional[str] = None


class PatientCreate(PatientBase):
    last_visit: Optional[date] = None
    notes: Optional[str] = None


class PatientListResponse(BaseModel):
    items: List[PatientSummary]
    total: int
    page: int
    page_size: int


class ImageMetadata(BaseModel):
    id: str
    patient_id: str
    type: str
    captured_at: datetime
    status: str
    storage_uri: Optional[str] = None
    public_url: Optional[str] = None


class ImageCreate(BaseModel):
    patient_id: str
    type: str
    captured_at: datetime
    auto_analyze: bool = True


class ImageUploadResponse(BaseModel):
    upload_url: str
    image: ImageMetadata
    auto_analyze: bool
    analysis_id: Optional[str] = None


class ModelConfigBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    description: Optional[str] = None
    model_type: Literal["cross_attn", "yolo_caries"] = "cross_attn"
    detector_path: str
    classifier_path: str
    detector_threshold: float = Field(0.25, ge=0, le=1)
    classification_threshold: float = Field(0.5, ge=0, le=1)
    max_teeth: int = Field(64, ge=1, le=256)


class ModelConfigCreate(ModelConfigBase):
    is_active: bool = False


class ModelConfigUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_type: Optional[Literal["cross_attn", "yolo_caries"]] = None
    detector_path: Optional[str] = None
    classifier_path: Optional[str] = None
    detector_threshold: Optional[float] = Field(None, ge=0, le=1)
    classification_threshold: Optional[float] = Field(None, ge=0, le=1)
    max_teeth: Optional[int] = Field(None, ge=1, le=256)
    is_active: Optional[bool] = None


class ModelConfigPublic(ModelConfigBase):
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class AnalysisBase(BaseModel):
    image_id: str
    requested_by: str
    status: str
    triggered_at: datetime
    completed_at: Optional[datetime] = None


class AnalysisSummary(AnalysisBase):
    id: str
    overall_assessment: Optional[str] = None
    preview: Optional[AnalysisPreview] = None


class AnalysisCreate(BaseModel):
    image_id: str
    requested_by: str
    priority: str = Field("standard", description="queued | high | urgent")


class FindingRegion(BaseModel):
    bbox: List[float] = Field(..., min_items=4, max_items=4)
    mask_uri: Optional[str] = None


class AnalysisFinding(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    finding_id: str
    type: str
    tooth_label: Optional[str] = None
    region: FindingRegion
    severity: str
    confidence: float = Field(..., ge=0, le=1)
    model_key: str
    model_version: str
    extra: dict = Field(default_factory=dict)
    note: Optional[str] = None
    confirmed: Optional[bool] = None

class AnalysisPreviewFinding(BaseModel):
    finding_id: str
    tooth_label: Optional[str] = None
    bbox: List[float] = Field(..., min_items=4, max_items=4)
    severity: str
    confidence: float
    assets: Optional[dict] = None
    bbox_normalized: Optional[List[float]] = None
    centroid: Optional[List[float]] = None
    color_bgr: Optional[List[int]] = None


class AnalysisPreview(BaseModel):
    image_uri: Optional[str] = None
    overlay_uri: Optional[str] = None
    image_size: Optional[List[int]] = None
    findings: List[AnalysisPreviewFinding] = Field(default_factory=list)



class PatientDetail(Patient):
    recent_images: List[ImageMetadata] = Field(default_factory=list)
    recent_analyses: List[AnalysisSummary] = Field(default_factory=list)


class AnalysisDetail(AnalysisSummary):
    patient: PatientSummary
    image: ImageMetadata
    findings: List[AnalysisFinding]
    detected_conditions: List[ConditionSummary]


class ReportAction(BaseModel):
    label: str
    description: str
    href: str


class AnalysisDetailResponse(AnalysisDetail):
    report_actions: List[ReportAction]


class SystemTimelineEvent(BaseModel):
    timestamp: datetime
    title: str
    description: str
    status: str


class AnalysisProgress(BaseModel):
    steps: List[SystemTimelineEvent]
    overall_confidence: float = Field(..., ge=0, le=1)


class AnalysisDetailExtended(AnalysisDetailResponse):
    progress: AnalysisProgress

