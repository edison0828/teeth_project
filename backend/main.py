"""FastAPI application serving Oral X-Ray analytics endpoints."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import distinct, func, or_, select
from sqlalchemy.orm import Session

from . import schemas
from .database import (
    Analysis,
    Finding,
    ImageStudy,
    Patient,
    SessionLocal,
    get_session,
    init_db,
)


UPLOAD_ROOT = Path(__file__).resolve().parent.parent / "uploaded_images"

app = FastAPI(title="Oral X-Ray Intelligence API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialise database and seed reference data."""

    init_db()
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as session:
        _seed_database(session)


# ---------------------------------------------------------------------------
# Data seeding utilities
# ---------------------------------------------------------------------------


def _seed_database(session: Session) -> None:
    """Populate the database with demo entities if empty."""

    existing_patients = session.scalar(select(func.count()).select_from(Patient))
    if existing_patients:
        return

    now = datetime.utcnow()

    patients = [
        Patient(
            id="P12345",
            name="Jane Doe",
            dob=date(1985, 3, 12),
            gender="female",
            contact="jane.doe@example.com",
            medical_history="Allergy: Penicillin. Previous root canal.",
            last_visit=date(2023, 10, 30),
            notes="Prefers SMS reminders.",
        ),
        Patient(
            id="P72631",
            name="John Smith",
            dob=date(1979, 11, 5),
            gender="male",
            contact="john.smith@example.com",
            medical_history="Hypertension. Regular cleanings.",
            last_visit=date(2023, 9, 14),
            notes=None,
        ),
        Patient(
            id="P55220",
            name="Emily Carter",
            dob=date(1992, 6, 2),
            gender="female",
            contact="emily.carter@example.com",
            medical_history="No known allergies.",
            last_visit=date(2023, 11, 7),
            notes="Pending orthodontic consultation.",
        ),
    ]
    session.add_all(patients)
    session.flush()

    images = [
        ImageStudy(
            id="IMG-001",
            patient_id="P12345",
            type="Panoramic",
            captured_at=now - timedelta(days=3),
            status="analyzed",
            storage_uri="s3://oral-xray/panoramic/P12345_20231030.png",
            original_filename="P12345_20231030.png",
            auto_analyze=True,
            created_at=now - timedelta(days=3),
        ),
        ImageStudy(
            id="IMG-002",
            patient_id="P72631",
            type="Bitewing",
            captured_at=now - timedelta(days=1, hours=2),
            status="pending_review",
            storage_uri="s3://oral-xray/bitewing/P72631_20231031.png",
            original_filename="P72631_20231031.png",
            auto_analyze=True,
            created_at=now - timedelta(days=1, hours=2),
        ),
        ImageStudy(
            id="IMG-003",
            patient_id="P55220",
            type="CBCT",
            captured_at=now - timedelta(days=6),
            status="queued",
            storage_uri="s3://oral-xray/cbct/P55220_20231026.dcm",
            original_filename="P55220_20231026.dcm",
            auto_analyze=True,
            created_at=now - timedelta(days=6),
        ),
    ]
    session.add_all(images)
    session.flush()

    analyses = [
        Analysis(
            id="AN-901",
            image_id="IMG-001",
            requested_by="Dr. Lee",
            status="completed",
            triggered_at=now - timedelta(days=3, hours=2),
            completed_at=now - timedelta(days=3),
            overall_assessment="Found 2 caries, 1 periodontal lesion.",
            detected_conditions=[
                {
                    "label": "Caries",
                    "count": 2,
                    "severity_breakdown": [
                        {"level": "moderate", "percentage": 0.6},
                        {"level": "severe", "percentage": 0.4},
                    ],
                },
                {
                    "label": "Periodontal",
                    "count": 1,
                    "severity_breakdown": [
                        {"level": "mild", "percentage": 1.0},
                    ],
                },
            ],
        ),
        Analysis(
            id="AN-902",
            image_id="IMG-002",
            requested_by="Dr. Lee",
            status="in_progress",
            triggered_at=now - timedelta(hours=5),
            completed_at=None,
            overall_assessment=None,
            detected_conditions=[],
        ),
    ]
    session.add_all(analyses)
    session.flush()

    findings = [
        Finding(
            analysis_id="AN-901",
            finding_id="FND-1",
            type="caries",
            tooth_label="FDI-26",
            region={"bbox": [240.0, 132.0, 88.0, 76.0], "mask_uri": None},
            severity="moderate",
            confidence=0.87,
            model_key="caries_detector",
            model_version="v1.2.0",
            extra={"distance_to_pulp": 1.4},
            note="Verify lesion depth",
            confirmed=True,
        ),
        Finding(
            analysis_id="AN-901",
            finding_id="FND-2",
            type="caries",
            tooth_label="FDI-27",
            region={"bbox": [320.0, 180.0, 72.0, 60.0], "mask_uri": None},
            severity="severe",
            confidence=0.92,
            model_key="caries_detector",
            model_version="v1.2.0",
            extra={"distance_to_pulp": 0.9},
            note=None,
            confirmed=None,
        ),
        Finding(
            analysis_id="AN-901",
            finding_id="FND-3",
            type="periodontal",
            tooth_label="FDI-36",
            region={"bbox": [180.0, 260.0, 110.0, 90.0], "mask_uri": None},
            severity="mild",
            confidence=0.74,
            model_key="periodontal_detector",
            model_version="v0.9.1",
            extra={"bone_loss_percentage": 0.18},
            note="Follow-up in 6 months",
            confirmed=False,
        ),
    ]
    session.add_all(findings)
    session.commit()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _generate_identifier(prefix: str, session: Session, model, length: int = 5) -> str:
    """Generate a unique identifier for the given model."""

    while True:
        candidate = f"{prefix}{uuid4().hex[:length].upper()}"
        if session.get(model, candidate) is None:
            return candidate


def _generate_patient_id(session: Session) -> str:
    return _generate_identifier("P", session, Patient, length=5)


def _generate_image_id(session: Session) -> str:
    return _generate_identifier("IMG-", session, ImageStudy, length=3)


def _generate_analysis_id(session: Session) -> str:
    return _generate_identifier("AN-", session, Analysis, length=3)


def _patient_summary(patient: Patient) -> schemas.PatientSummary:
    latest_image = max(patient.images, key=lambda img: img.captured_at, default=None)
    return schemas.PatientSummary(
        id=patient.id,
        name=patient.name,
        last_visit=patient.last_visit,
        most_recent_study=latest_image.type if latest_image else None,
    )


def _image_metadata(image: ImageStudy) -> schemas.ImageMetadata:
    return schemas.ImageMetadata(
        id=image.id,
        patient_id=image.patient_id,
        type=image.type,
        captured_at=image.captured_at,
        status=image.status,
        storage_uri=image.storage_uri,
    )


def _analysis_summary(analysis: Analysis) -> schemas.AnalysisSummary:
    return schemas.AnalysisSummary(
        id=analysis.id,
        image_id=analysis.image_id,
        requested_by=analysis.requested_by,
        status=analysis.status,
        triggered_at=analysis.triggered_at,
        completed_at=analysis.completed_at,
        overall_assessment=analysis.overall_assessment,
    )


def _condition_summaries(raw_conditions: List[Dict]) -> List[schemas.ConditionSummary]:
    summaries: List[schemas.ConditionSummary] = []
    for condition in raw_conditions or []:
        severities = [
            schemas.SeveritySlice(level=s["level"], percentage=s["percentage"])
            for s in condition.get("severity_breakdown", [])
        ]
        summaries.append(
            schemas.ConditionSummary(
                label=condition.get("label", "Unknown"),
                count=condition.get("count", 0),
                severity_breakdown=severities,
            )
        )
    return summaries


def _analysis_detail_from_instance(analysis: Analysis) -> schemas.AnalysisDetailExtended:
    image = analysis.image
    if image is None or image.patient is None:
        raise HTTPException(status_code=500, detail="Analysis is missing related image or patient information")

    patient_summary = _patient_summary(image.patient)
    image_metadata = _image_metadata(image)
    findings = [
        schemas.AnalysisFinding(
            finding_id=f.finding_id,
            type=f.type,
            tooth_label=f.tooth_label,
            region=f.region or {"bbox": [], "mask_uri": None},
            severity=f.severity or "",
            confidence=f.confidence or 0.0,
            model_key=f.model_key or "",
            model_version=f.model_version or "",
            extra=f.extra or {},
            note=f.note,
            confirmed=f.confirmed,
        )
        for f in analysis.findings
    ]

    detected_conditions = _condition_summaries(analysis.detected_conditions)

    progress_steps = [
        schemas.SystemTimelineEvent(
            timestamp=analysis.triggered_at,
            title="Upload received",
            description="Image queued for preprocessing",
            status="done",
        ),
        schemas.SystemTimelineEvent(
            timestamp=analysis.triggered_at + timedelta(minutes=5),
            title="Preprocessing",
            description="Contrast normalization and orientation alignment",
            status="done",
        ),
        schemas.SystemTimelineEvent(
            timestamp=(analysis.triggered_at + timedelta(minutes=15)),
            title="AI models",
            description="Running caries / periodontal detectors",
            status="done" if analysis.status == "completed" else "active",
        ),
        schemas.SystemTimelineEvent(
            timestamp=analysis.completed_at or datetime.utcnow(),
            title="Report generation",
            description="Formatting consolidated findings",
            status="done" if analysis.status == "completed" else "pending",
        ),
    ]

    return schemas.AnalysisDetailExtended(
        id=analysis.id,
        image_id=analysis.image_id,
        requested_by=analysis.requested_by,
        status=analysis.status,
        triggered_at=analysis.triggered_at,
        completed_at=analysis.completed_at,
        overall_assessment=analysis.overall_assessment,
        patient=patient_summary,
        image=image_metadata,
        findings=findings,
        detected_conditions=detected_conditions,
        report_actions=[
            schemas.ReportAction(
                label="Generate Report",
                description="Export PDF clinical report",
                href=f"/reports/{analysis.id}?format=pdf",
            ),
            schemas.ReportAction(
                label="Download CSV",
                description="Download raw findings data",
                href=f"/reports/{analysis.id}?format=csv",
            ),
        ],
        progress=schemas.AnalysisProgress(
            overall_confidence=0.86,
            steps=progress_steps,
        ),
    )


def _analysis_detail(session: Session, analysis_id: str) -> schemas.AnalysisDetailExtended:
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _analysis_detail_from_instance(analysis)


def _enqueue_analysis(
    session: Session,
    *,
    image: ImageStudy,
    requested_by: str,
    priority: str = "standard",
) -> Analysis:
    analysis_id = _generate_analysis_id(session)
    status = "queued" if priority == "standard" else "scheduled"
    analysis = Analysis(
        id=analysis_id,
        image_id=image.id,
        requested_by=requested_by,
        status=status,
        priority=priority,
        triggered_at=datetime.utcnow(),
        completed_at=None,
        overall_assessment=None,
        detected_conditions=[],
    )
    session.add(analysis)
    session.flush()
    return analysis


def _create_image_record(
    session: Session,
    *,
    patient_id: str,
    study_type: str,
    captured_at: datetime,
    auto_analyze: bool,
    storage_uri: str | None,
    original_filename: str | None,
    notes: str | None = None,
    status: str | None = None,
    image_id: str | None = None,
    priority: str = "standard",
    requested_by: str = "system",
) -> ImageStudy:
    if image_id is None:
        image_id = _generate_image_id(session)
    record_status = status or ("queued" if auto_analyze else "uploaded")

    image = ImageStudy(
        id=image_id,
        patient_id=patient_id,
        type=study_type,
        captured_at=captured_at,
        status=record_status,
        storage_uri=storage_uri,
        original_filename=original_filename,
        notes=notes,
        auto_analyze=auto_analyze,
        created_at=datetime.utcnow(),
    )
    session.add(image)
    session.flush()

    if auto_analyze:
        _enqueue_analysis(session, image=image, requested_by=requested_by, priority=priority)

    return image


async def _persist_upload(file: UploadFile, destination: Path) -> str:
    """Persist the uploaded file and return its storage URI."""

    contents = await file.read()
    destination.write_bytes(contents)
    return destination.as_posix()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/dashboard/overview", response_model=schemas.DashboardOverview)
async def get_dashboard_overview(db: Session = Depends(get_session)) -> schemas.DashboardOverview:
    patients = db.scalars(select(Patient)).all()
    recent_patients = sorted(
        (_patient_summary(patient) for patient in patients),
        key=lambda summary: summary.last_visit or date.min,
        reverse=True,
    )[:5]

    pending_images_query = select(ImageStudy).where(ImageStudy.status.in_(["queued", "pending_review"]))
    pending_images_models = db.scalars(pending_images_query).all()
    pending_images = [
        schemas.ImageQueueItem(
            id=image.id,
            patient_id=image.patient_id,
            patient_name=image.patient.name if image.patient else "",
            image_type=image.type,
            submitted_at=image.created_at or image.captured_at,
            status=image.status,
        )
        for image in pending_images_models
    ]

    now = datetime.utcnow()
    analyses = db.scalars(select(Analysis)).all()
    new_reports = sum(1 for analysis in analyses if analysis.status == "completed")

    week_ago = now - timedelta(days=7)
    weekly_volume = sum(1 for analysis in analyses if analysis.triggered_at >= week_ago)

    detailed_conditions: Dict[str, Dict[str, object]] = {}
    for analysis in analyses:
        for condition in analysis.detected_conditions or []:
            entry = detailed_conditions.setdefault(
                condition.get("label", "Unknown"),
                {"count": 0, "severity_breakdown": condition.get("severity_breakdown", [])},
            )
            entry["count"] += condition.get("count", 0)

    condition_summaries = [
        schemas.ConditionSummary(
            label=label,
            count=data["count"],
            severity_breakdown=[
                schemas.SeveritySlice(level=item["level"], percentage=item["percentage"])
                for item in data.get("severity_breakdown", [])
            ],
        )
        for label, data in detailed_conditions.items()
    ]

    completion_times = [
        (analysis.completed_at - analysis.triggered_at).total_seconds() / 60
        for analysis in analyses
        if analysis.completed_at
    ]
    average_processing = sum(completion_times) / len(completion_times) if completion_times else 0.0

    detection_denominator = len(analyses)
    analyses_with_findings = db.scalar(
        select(func.count(distinct(Finding.analysis_id)))
    ) or 0
    detection_rate = (analyses_with_findings / detection_denominator) if detection_denominator else 0.0

    return schemas.DashboardOverview(
        quick_actions=[
            schemas.QuickAction(
                id="upload",
                label="Upload New Image",
                description="Drag and drop radiographs for preprocessing",
                action="/image-upload",
            ),
            schemas.QuickAction(
                id="search",
                label="Search Patients",
                description="Find records by name, MRN or date",
                action="/patients",
            ),
        ],
        system_status=schemas.SystemStatus(
            pending_images=len(pending_images),
            new_reports=new_reports,
            models_active=4,
            last_synced=now,
        ),
        statistics=schemas.StatisticsOverview(
            weekly_volume=weekly_volume,
            week_over_week_change=0.12,
            detection_rate=detection_rate,
            average_processing_time=round(average_processing, 2),
            uptime_percentage=0.995,
        ),
        detected_conditions=condition_summaries,
        recent_patients=recent_patients,
        pending_images=pending_images,
    )


@app.get("/api/patients", response_model=schemas.PatientListResponse)
async def list_patients(
    search: str | None = None,
    page: int = 1,
    page_size: int = 10,
    db: Session = Depends(get_session),
) -> schemas.PatientListResponse:
    stmt = select(Patient)
    if search:
        lowered = f"%{search.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(Patient.name).like(lowered),
                func.lower(Patient.id).like(lowered),
            )
        )
    patients = db.scalars(stmt).all()
    total = len(patients)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = patients[start:end]
    summaries = [_patient_summary(patient) for patient in page_items]
    return schemas.PatientListResponse(
        items=summaries,
        total=total,
        page=page,
        page_size=page_size,
    )


@app.post("/api/patients", response_model=schemas.Patient)
async def create_patient(payload: schemas.PatientCreate, db: Session = Depends(get_session)) -> schemas.Patient:
    patient_id = _generate_patient_id(db)
    patient = Patient(
        id=patient_id,
        name=payload.name,
        dob=payload.dob,
        gender=payload.gender,
        contact=payload.contact,
        medical_history=payload.medical_history,
        last_visit=payload.last_visit,
        notes=payload.notes,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return schemas.Patient(
        id=patient.id,
        name=patient.name,
        dob=patient.dob,
        gender=patient.gender,
        contact=patient.contact,
        medical_history=patient.medical_history,
        last_visit=patient.last_visit,
        notes=patient.notes,
    )


@app.get("/api/patients/{patient_id}", response_model=schemas.PatientDetail)
async def get_patient(patient_id: str, db: Session = Depends(get_session)) -> schemas.PatientDetail:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    recent_images = sorted(patient.images, key=lambda img: img.captured_at, reverse=True)
    analyses_map: Dict[str, Analysis] = {}
    for image in patient.images:
        for analysis in image.analyses:
            analyses_map[analysis.id] = analysis
    patient_analyses = sorted(analyses_map.values(), key=lambda a: a.triggered_at, reverse=True)

    return schemas.PatientDetail(
        id=patient.id,
        name=patient.name,
        dob=patient.dob,
        gender=patient.gender,
        contact=patient.contact,
        medical_history=patient.medical_history,
        last_visit=patient.last_visit,
        notes=patient.notes,
        recent_images=[_image_metadata(image) for image in recent_images],
        recent_analyses=[_analysis_summary(analysis) for analysis in patient_analyses],
    )


@app.get(
    "/api/patients/{patient_id}/analyses",
    response_model=List[schemas.AnalysisSummary],
)
async def get_patient_analyses(patient_id: str, db: Session = Depends(get_session)) -> List[schemas.AnalysisSummary]:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    analyses: List[Analysis] = []
    for image in patient.images:
        analyses.extend(image.analyses)

    return [_analysis_summary(analysis) for analysis in sorted(analyses, key=lambda a: a.triggered_at, reverse=True)]


@app.post("/api/images", response_model=schemas.ImageUploadResponse)
async def create_image(payload: schemas.ImageCreate, db: Session = Depends(get_session)) -> schemas.ImageUploadResponse:
    patient = db.get(Patient, payload.patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    image = _create_image_record(
        db,
        patient_id=payload.patient_id,
        study_type=payload.type,
        captured_at=payload.captured_at,
        auto_analyze=payload.auto_analyze,
        storage_uri=None,
        original_filename=None,
        notes=None,
        priority="standard",
    )
    db.commit()
    db.refresh(image)

    return schemas.ImageUploadResponse(
        upload_url=f"/uploaded_images/{image.id}",
        image=_image_metadata(image),
        auto_analyze=payload.auto_analyze,
    )


@app.post(
    "/api/uploads/images",
    response_model=schemas.ImageUploadResponse,
    status_code=201,
)
async def upload_image(
    patient_id: str = Form(...),
    study_type: str = Form(..., alias="type"),
    captured_at: str = Form(...),
    auto_analyze: bool = Form(True),
    priority: str = Form("standard"),
    notes: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
) -> schemas.ImageUploadResponse:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        captured_dt = datetime.fromisoformat(captured_at)
    except ValueError as exc:  # pragma: no cover - validation guard
        raise HTTPException(status_code=422, detail="Invalid captured_at format. Use ISO 8601.") from exc

    image_id = _generate_image_id(db)
    extension = Path(file.filename).suffix or ""
    destination = UPLOAD_ROOT / f"{image_id}{extension}"
    storage_uri = await _persist_upload(file, destination)
    notes_value = notes.strip() if notes and notes.strip() else None

    image = _create_image_record(
        db,
        patient_id=patient_id,
        study_type=study_type,
        captured_at=captured_dt,
        auto_analyze=auto_analyze,
        storage_uri=storage_uri,
        original_filename=file.filename,
        notes=notes_value,
        priority=priority,
        image_id=image_id,
    )
    db.commit()
    db.refresh(image)

    return schemas.ImageUploadResponse(
        upload_url=storage_uri,
        image=_image_metadata(image),
        auto_analyze=auto_analyze,
    )


@app.post("/api/analyses", response_model=schemas.AnalysisDetailExtended, status_code=201)
async def create_analysis(
    payload: schemas.AnalysisCreate,
    db: Session = Depends(get_session),
) -> schemas.AnalysisDetailExtended:
    image = db.get(ImageStudy, payload.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    analysis = _enqueue_analysis(
        db,
        image=image,
        requested_by=payload.requested_by,
        priority=payload.priority,
    )
    db.commit()
    db.refresh(analysis)

    return _analysis_detail_from_instance(analysis)


@app.get("/api/analyses/{analysis_id}", response_model=schemas.AnalysisDetailExtended)
async def get_analysis(analysis_id: str, db: Session = Depends(get_session)) -> schemas.AnalysisDetailExtended:
    return _analysis_detail(db, analysis_id)


@app.get("/api/analyses", response_model=List[schemas.AnalysisSummary])
async def list_analyses(
    status: str | None = None,
    db: Session = Depends(get_session),
) -> List[schemas.AnalysisSummary]:
    stmt = select(Analysis)
    if status:
        stmt = stmt.where(Analysis.status == status)
    analyses = db.scalars(stmt).all()
    analyses_sorted = sorted(analyses, key=lambda a: a.triggered_at, reverse=True)
    return [_analysis_summary(analysis) for analysis in analyses_sorted]



