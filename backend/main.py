"""FastAPI application serving Oral X-Ray analytics endpoints."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import schemas

app = FastAPI(title="Oral X-Ray Intelligence API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory data store used to illustrate the system design proposal.
# ---------------------------------------------------------------------------
_now = datetime.utcnow()

patients_db: Dict[str, Dict] = {
    "P12345": {
        "id": "P12345",
        "name": "Jane Doe",
        "dob": date(1985, 3, 12),
        "gender": "female",
        "contact": "jane.doe@example.com",
        "medical_history": "Allergy: Penicillin. Previous root canal.",
        "last_visit": date(2023, 10, 30),
        "notes": "Prefers SMS reminders.",
    },
    "P72631": {
        "id": "P72631",
        "name": "John Smith",
        "dob": date(1979, 11, 5),
        "gender": "male",
        "contact": "john.smith@example.com",
        "medical_history": "Hypertension. Regular cleanings.",
        "last_visit": date(2023, 9, 14),
        "notes": None,
    },
    "P55220": {
        "id": "P55220",
        "name": "Emily Carter",
        "dob": date(1992, 6, 2),
        "gender": "female",
        "contact": "emily.carter@example.com",
        "medical_history": "No known allergies.",
        "last_visit": date(2023, 11, 7),
        "notes": "Pending orthodontic consultation.",
    },
}

images_db: Dict[str, Dict] = {
    "IMG-001": {
        "id": "IMG-001",
        "patient_id": "P12345",
        "type": "Panoramic",
        "captured_at": _now - timedelta(days=3),
        "status": "analyzed",
        "storage_uri": "s3://oral-xray/panoramic/P12345_20231030.png",
    },
    "IMG-002": {
        "id": "IMG-002",
        "patient_id": "P72631",
        "type": "Bitewing",
        "captured_at": _now - timedelta(days=1, hours=2),
        "status": "pending_review",
        "storage_uri": "s3://oral-xray/bitewing/P72631_20231031.png",
    },
    "IMG-003": {
        "id": "IMG-003",
        "patient_id": "P55220",
        "type": "CBCT",
        "captured_at": _now - timedelta(days=6),
        "status": "queued",
        "storage_uri": "s3://oral-xray/cbct/P55220_20231026.dcm",
    },
}

analyses_db: Dict[str, Dict] = {
    "AN-901": {
        "id": "AN-901",
        "image_id": "IMG-001",
        "requested_by": "Dr. Lee",
        "status": "completed",
        "triggered_at": _now - timedelta(days=3, hours=2),
        "completed_at": _now - timedelta(days=3),
        "overall_assessment": "Found 2 caries, 1 periodontal lesion.",
        "detected_conditions": [
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
    },
    "AN-902": {
        "id": "AN-902",
        "image_id": "IMG-002",
        "requested_by": "Dr. Lee",
        "status": "in_progress",
        "triggered_at": _now - timedelta(hours=5),
        "completed_at": None,
        "overall_assessment": None,
        "detected_conditions": [],
    },
}

findings_db: Dict[str, List[Dict]] = {
    "AN-901": [
        {
            "finding_id": "FND-1",
            "type": "caries",
            "tooth_label": "FDI-26",
            "region": {"bbox": [240.0, 132.0, 88.0, 76.0], "mask_uri": None},
            "severity": "moderate",
            "confidence": 0.87,
            "model_key": "caries_detector",
            "model_version": "v1.2.0",
            "extra": {"distance_to_pulp": 1.4},
            "note": "Verify lesion depth",
            "confirmed": True,
        },
        {
            "finding_id": "FND-2",
            "type": "caries",
            "tooth_label": "FDI-27",
            "region": {"bbox": [320.0, 180.0, 72.0, 60.0], "mask_uri": None},
            "severity": "severe",
            "confidence": 0.92,
            "model_key": "caries_detector",
            "model_version": "v1.2.0",
            "extra": {"distance_to_pulp": 0.9},
            "note": None,
            "confirmed": None,
        },
        {
            "finding_id": "FND-3",
            "type": "periodontal",
            "tooth_label": "FDI-36",
            "region": {"bbox": [180.0, 260.0, 110.0, 90.0], "mask_uri": None},
            "severity": "mild",
            "confidence": 0.74,
            "model_key": "periodontal_detector",
            "model_version": "v0.9.1",
            "extra": {"bone_loss_percentage": 0.18},
            "note": "Follow-up in 6 months",
            "confirmed": False,
        },
    ]
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _patient_summary(patient: Dict) -> schemas.PatientSummary:
    image = next((img for img in images_db.values() if img["patient_id"] == patient["id"]), None)
    study = image["type"] if image else None
    return schemas.PatientSummary(
        id=patient["id"],
        name=patient["name"],
        last_visit=patient.get("last_visit"),
        most_recent_study=study,
    )


def _image_metadata(image: Dict) -> schemas.ImageMetadata:
    return schemas.ImageMetadata(
        id=image["id"],
        patient_id=image["patient_id"],
        type=image["type"],
        captured_at=image["captured_at"],
        status=image["status"],
        storage_uri=image.get("storage_uri"),
    )


def _analysis_summary(analysis: Dict) -> schemas.AnalysisSummary:
    return schemas.AnalysisSummary(
        id=analysis["id"],
        image_id=analysis["image_id"],
        requested_by=analysis["requested_by"],
        status=analysis["status"],
        triggered_at=analysis["triggered_at"],
        completed_at=analysis.get("completed_at"),
        overall_assessment=analysis.get("overall_assessment"),
    )


def _analysis_detail(analysis_id: str) -> schemas.AnalysisDetailExtended:
    if analysis_id not in analyses_db:
        raise HTTPException(status_code=404, detail="Analysis not found")
    analysis = analyses_db[analysis_id]
    image = images_db[analysis["image_id"]]
    patient = patients_db[image["patient_id"]]

    findings = [schemas.AnalysisFinding(**f) for f in findings_db.get(analysis_id, [])]
    detected_conditions = [
        schemas.ConditionSummary(**condition)
        for condition in analysis.get("detected_conditions", [])
    ]

    detail = schemas.AnalysisDetailExtended(
        id=analysis["id"],
        image_id=analysis["image_id"],
        requested_by=analysis["requested_by"],
        status=analysis["status"],
        triggered_at=analysis["triggered_at"],
        completed_at=analysis.get("completed_at"),
        overall_assessment=analysis.get("overall_assessment"),
        patient=_patient_summary(patient),
        image=_image_metadata(image),
        findings=findings,
        detected_conditions=detected_conditions,
        report_actions=[
            schemas.ReportAction(
                label="Generate Report",
                description="Export PDF clinical report",
                href=f"/reports/{analysis_id}?format=pdf",
            ),
            schemas.ReportAction(
                label="Download CSV",
                description="Download raw findings data",
                href=f"/reports/{analysis_id}?format=csv",
            ),
        ],
        progress=schemas.AnalysisProgress(
            overall_confidence=0.86,
            steps=[
                schemas.SystemTimelineEvent(
                    timestamp=analysis["triggered_at"],
                    title="Upload received",
                    description="Image queued for preprocessing",
                    status="done",
                ),
                schemas.SystemTimelineEvent(
                    timestamp=analysis["triggered_at"] + timedelta(minutes=5),
                    title="Preprocessing",
                    description="Contrast normalization and orientation alignment",
                    status="done",
                ),
                schemas.SystemTimelineEvent(
                    timestamp=analysis["triggered_at"] + timedelta(minutes=15),
                    title="AI models",
                    description="Running caries / periodontal detectors",
                    status="active" if analysis["status"] != "completed" else "done",
                ),
                schemas.SystemTimelineEvent(
                    timestamp=analysis.get("completed_at", _now),
                    title="Report generation",
                    description="Formatting consolidated findings",
                    status="pending" if analysis["status"] != "completed" else "done",
                ),
            ],
        ),
    )
    return detail


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/dashboard/overview", response_model=schemas.DashboardOverview)
async def get_dashboard_overview() -> schemas.DashboardOverview:
    recent_patients = sorted(
        [_patient_summary(p) for p in patients_db.values()],
        key=lambda p: p.last_visit or date.min,
        reverse=True,
    )[:5]

    pending_images = [
        schemas.ImageQueueItem(
            id=image["id"],
            patient_id=image["patient_id"],
            patient_name=patients_db[image["patient_id"]]["name"],
            image_type=image["type"],
            submitted_at=image["captured_at"],
            status=image["status"],
        )
        for image in images_db.values()
        if image["status"] in {"queued", "pending_review"}
    ]

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
            new_reports=sum(1 for a in analyses_db.values() if a["status"] == "completed"),
            models_active=4,
            last_synced=_now,
        ),
        statistics=schemas.StatisticsOverview(
            weekly_volume=42,
            week_over_week_change=0.12,
            detection_rate=0.86,
            average_processing_time=6.5,
            uptime_percentage=0.995,
        ),
        detected_conditions=[
            schemas.ConditionSummary(
                label="Caries",
                count=18,
                severity_breakdown=[
                    schemas.SeveritySlice(level="mild", percentage=0.35),
                    schemas.SeveritySlice(level="moderate", percentage=0.45),
                    schemas.SeveritySlice(level="severe", percentage=0.2),
                ],
            ),
            schemas.ConditionSummary(
                label="Periodontal",
                count=9,
                severity_breakdown=[
                    schemas.SeveritySlice(level="mild", percentage=0.5),
                    schemas.SeveritySlice(level="moderate", percentage=0.3),
                    schemas.SeveritySlice(level="severe", percentage=0.2),
                ],
            ),
            schemas.ConditionSummary(
                label="Periapical",
                count=5,
                severity_breakdown=[
                    schemas.SeveritySlice(level="mild", percentage=0.2),
                    schemas.SeveritySlice(level="moderate", percentage=0.6),
                    schemas.SeveritySlice(level="severe", percentage=0.2),
                ],
            ),
        ],
        recent_patients=recent_patients,
        pending_images=pending_images,
    )


@app.get("/api/patients", response_model=schemas.PatientListResponse)
async def list_patients(
    search: str | None = None, page: int = 1, page_size: int = 10
) -> schemas.PatientListResponse:
    items = list(patients_db.values())
    if search:
        lowered = search.lower()
        items = [
            patient
            for patient in items
            if lowered in patient["name"].lower() or lowered in patient["id"].lower()
        ]

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    summaries = [_patient_summary(patient) for patient in page_items]
    return schemas.PatientListResponse(
        items=summaries,
        total=total,
        page=page,
        page_size=page_size,
    )


@app.post("/api/patients", response_model=schemas.Patient)
async def create_patient(payload: schemas.PatientCreate) -> schemas.Patient:
    patient_id = f"P{uuid4().hex[:5].upper()}"
    patient = payload.model_dump()
    patient.update({"id": patient_id})
    patients_db[patient_id] = patient
    return schemas.Patient(**patient)


@app.get("/api/patients/{patient_id}", response_model=schemas.PatientDetail)
async def get_patient(patient_id: str) -> schemas.PatientDetail:
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient = patients_db[patient_id]
    patient_images = [
        _image_metadata(image)
        for image in images_db.values()
        if image["patient_id"] == patient_id
    ]
    patient_analyses = [
        _analysis_summary(analysis)
        for analysis in analyses_db.values()
        if images_db[analysis["image_id"]]["patient_id"] == patient_id
    ]
    return schemas.PatientDetail(
        **patient,
        recent_images=patient_images,
        recent_analyses=patient_analyses,
    )


@app.get(
    "/api/patients/{patient_id}/analyses", response_model=List[schemas.AnalysisSummary]
)
async def get_patient_analyses(patient_id: str) -> List[schemas.AnalysisSummary]:
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")

    analyses = [
        _analysis_summary(analysis)
        for analysis in analyses_db.values()
        if images_db[analysis["image_id"]]["patient_id"] == patient_id
    ]
    return analyses


@app.post("/api/images", response_model=schemas.ImageUploadResponse)
async def create_image(payload: schemas.ImageCreate) -> schemas.ImageUploadResponse:
    if payload.patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")

    image_id = f"IMG-{uuid4().hex[:3].upper()}"
    image = {
        "id": image_id,
        "patient_id": payload.patient_id,
        "type": payload.type,
        "captured_at": payload.captured_at,
        "status": "queued" if payload.auto_analyze else "uploaded",
        "storage_uri": f"s3://oral-xray/uploads/{image_id}.dcm",
    }
    images_db[image_id] = image

    upload_url = f"https://storage.local/upload/{image_id}"

    if payload.auto_analyze:
        analysis_id = f"AN-{uuid4().hex[:3].upper()}"
        analyses_db[analysis_id] = {
            "id": analysis_id,
            "image_id": image_id,
            "requested_by": "system",
            "status": "queued",
            "triggered_at": datetime.utcnow(),
            "completed_at": None,
            "overall_assessment": None,
            "detected_conditions": [],
        }

    return schemas.ImageUploadResponse(
        upload_url=upload_url,
        image=_image_metadata(image),
        auto_analyze=payload.auto_analyze,
    )


@app.post("/api/analyses", response_model=schemas.AnalysisDetailExtended, status_code=201)
async def create_analysis(
    payload: schemas.AnalysisCreate,
) -> schemas.AnalysisDetailResponse:
    if payload.image_id not in images_db:
        raise HTTPException(status_code=404, detail="Image not found")

    analysis_id = f"AN-{uuid4().hex[:3].upper()}"
    analysis = {
        "id": analysis_id,
        "image_id": payload.image_id,
        "requested_by": payload.requested_by,
        "status": "queued" if payload.priority == "standard" else "scheduled",
        "triggered_at": datetime.utcnow(),
        "completed_at": None,
        "overall_assessment": None,
        "detected_conditions": [],
    }
    analyses_db[analysis_id] = analysis
    findings_db.setdefault(analysis_id, [])
    return _analysis_detail(analysis_id)


@app.get("/api/analyses/{analysis_id}", response_model=schemas.AnalysisDetailExtended)
async def get_analysis(analysis_id: str) -> schemas.AnalysisDetailExtended:
    return _analysis_detail(analysis_id)


@app.get("/api/analyses", response_model=List[schemas.AnalysisSummary])
async def list_analyses(status: str | None = None) -> List[schemas.AnalysisSummary]:
    analyses = analyses_db.values()
    if status:
        analyses = [a for a in analyses if a["status"] == status]
    return [_analysis_summary(analysis) for analysis in analyses]
