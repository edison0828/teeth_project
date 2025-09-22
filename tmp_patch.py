from pathlib import Path
import re
path = Path(''backend/main.py'')
text = path.read_text()

text = text.replace('from typing import Dict, List, Optional\n', 'from typing import Dict, List, Optional, Tuple\n')
text = text.replace('from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile\n', 'from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile\n')
text = text.replace('    Patient,\n    SessionLocal,\n    get_session,\n    init_db,\n)\n\n\nUPLOAD_ROOT', '    Patient,\n    SessionLocal,\n    ModelConfig,\n    get_session,\n    init_db,\n)\n\n\nfrom .analysis_service import process_analysis_job\n\n\nUPLOAD_ROOT')

seed_pattern = '    existing_patients = session.scalar(select(func.count()).select_from(Patient))'
seed_replacement = '''    existing_models = session.scalar(select(func.count()).select_from(ModelConfig))
    if not existing_models:
        models_root = Path(__file__).resolve().parent.parent / "models"
        default_detector = (models_root / "fdi_all seg.pt").as_posix()
        default_classifier = (models_root / "cross_attn_fdi_camAlignA.pth").as_posix()
        session.add(
            ModelConfig(
                id=_generate_model_config_id(session),
                name="Cross-Attn Caries",
                description="Cross-attention classifier with YOLO tooth detector",
                detector_path=default_detector,
                classifier_path=default_classifier,
                detector_threshold=0.25,
                classification_threshold=0.5,
                max_teeth=64,
                is_active=True,
            )
        )
        session.commit()

    existing_patients = session.scalar(select(func.count()).select_from(Patient))'''
text = text.replace(seed_pattern, seed_replacement, 1)

image_metadata_block = '''    return schemas.ImageMetadata(
        id=image.id,
        patient_id=image.patient_id,
        type=image.type,
        captured_at=image.captured_at,
        status=image.status,
        storage_uri=image.storage_uri,
    )

'''
helpers = '''    return schemas.ImageMetadata(
        id=image.id,
        patient_id=image.patient_id,
        type=image.type,
        captured_at=image.captured_at,
        status=image.status,
        storage_uri=image.storage_uri,
    )


def _model_config_public(model: ModelConfig) -> schemas.ModelConfigPublic:
    return schemas.ModelConfigPublic(
        id=model.id,
        name=model.name,
        description=model.description,
        detector_path=model.detector_path,
        classifier_path=model.classifier_path,
        detector_threshold=model.detector_threshold,
        classification_threshold=model.classification_threshold,
        max_teeth=model.max_teeth,
        is_active=model.is_active,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _active_model(session: Session) -> Optional[ModelConfig]:
    return session.scalars(select(ModelConfig).where(ModelConfig.is_active == True)).first()

'''
text = text.replace(image_metadata_block, helpers, 1)

text = text.replace('def _generate_analysis_id(session: Session) -> str\n    return _generate_identifier("AN-", session, Analysis, length=3)\n\n', 'def _generate_analysis_id(session: Session) -> str\n    return _generate_identifier("AN-", session, Analysis, length=3)\n\n\ndef _generate_model_config_id(session: Session) -> str\n    return _generate_identifier("MODEL-", session, ModelConfig, length=4)\n\n')

findings_pattern = r'(    findings = \[\n        schemas.AnalysisFinding\([\s\S]*?\n    \]\n\n)    detected_conditions'
findings_replacement = r"\1    confidences = [finding.confidence for finding in findings if finding.confidence is not None]\n    overall_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0\n\n    detected_conditions"
text = re.sub(findings_pattern, findings_replacement, text, count=1)

text = text.replace('overall_confidence=0.86', 'overall_confidence=overall_confidence')

text = text.replace(') -> ImageStudy:', ') -> Tuple[ImageStudy, Optional[Analysis]]:')
text = text.replace('    record_status = status or ("queued" if auto_analyze else "uploaded")\n\n    image = ImageStudy(', '    record_status = status or ("queued" if auto_analyze else "uploaded")\n    analysis: Analysis | None = None\n\n    image = ImageStudy(')
text = text.replace('    if auto_analyze:\n        _enqueue_analysis(session, image=image, requested_by=requested_by, priority=priority)\n\n    return image', '    if auto_analyze:\n        analysis = _enqueue_analysis(session, image=image, requested_by=requested_by, priority=priority)\n\n    return image, analysis')

week_pattern = '    week_ago = now - timedelta(days=7)\n    weekly_volume = sum(1 for analysis in analyses if analysis.triggered_at >= week_ago)\n\n    detailed_conditions'
week_replacement = '    week_ago = now - timedelta(days=7)\n    weekly_volume = sum(1 for analysis in analyses if analysis.triggered_at >= week_ago)\n\n    model_configs = db.scalars(select(ModelConfig)).all()\n    active_model = next((model for model in model_configs if model.is_active), None)\n    models_active = sum(1 for model in model_configs if model.is_active)\n\n    detailed_conditions'
text = text.replace(week_pattern, week_replacement, 1)
text = text.replace('models_active=4,\n            last_synced=now,', 'models_active=models_active,\n            active_model_name=active_model.name if active_model else None,\n            last_synced=now,')

create_image_block = '@app.post("/api/images", response_model=schemas.ImageUploadResponse)\nasync def create_image(\n    payload: schemas.ImageCreate,\n    db: Session = Depends(get_session),\n    current_user: User = Depends(get_current_active_user),\n) -> schemas.ImageUploadResponse:\n    patient = db.get(Patient, payload.patient_id)\n    if patient is None:\n        raise HTTPException(status_code=404, detail="Patient not found")\n\n    image = _create_image_record(\n        db,\n        patient_id=payload.patient_id,\n        study_type=payload.type,\n        captured_at=payload.captured_at,\n        auto_analyze=payload.auto_analyze,\n        storage_uri=None,\n        original_filename=None,\n        notes=None,\n        priority="standard",\n        requested_by=current_user.full_name or current_user.email,\n    )\n    db.commit()\n    db.refresh(image)\n\n    return schemas.ImageUploadResponse(\n        upload_url=f"/uploaded images/{image.id}",\n        image=_image_metadata(image),\n        auto_analyze=payload.auto_analyze,\n    )\n\n'
create_image_new = '@app.post("/api/images", response_model=schemas.ImageUploadResponse)\nasync def create_image(\n    payload: schemas.ImageCreate,\n    background_tasks: BackgroundTasks,\n    db: Session = Depends(get_session),\n    current_user: User = Depends(get_current_active_user),\n) -> schemas.ImageUploadResponse:\n    patient = db.get(Patient, payload.patient_id)\n    if patient is None:\n        raise HTTPException(status_code=404, detail="Patient not found")\n\n    image, analysis = _create_image_record(\n        db,\n        patient_id=payload.patient_id,\n        study_type=payload.type,\n        captured_at=payload.captured_at,\n        auto_analyze=payload.auto_analyze,\n        storage_uri=None,\n        original_filename=None,\n        notes=None,\n        priority="standard",\n        requested_by=current_user.full_name or current_user.email,\n    )\n    db.commit()\n    db.refresh(image)\n    if analysis is not None:\n        db.refresh(analysis)\n        background_tasks.add_task(process_analysis_job, analysis.id)\n\n    return schemas.ImageUploadResponse(\n        upload_url=f"/uploaded_images/{image.id}",\n        image=_image_metadata(image),\n        auto_analyze=payload.auto_analyze,\n        analysis_id=analysis.id if analysis else None,\n    )\n\n'
text = text.replace(create_image_block, create_image_new, 1)

d
