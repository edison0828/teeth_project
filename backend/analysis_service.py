"""負責執行影像分析並回寫資料庫。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from sqlalchemy import select

from sqlalchemy.orm import Session

from .database import Analysis, Finding, ImageStudy, ModelConfig, SessionLocal
from .inference import InferenceError, ModelDefinition, get_inference_engine


def _reset_findings(session: Session, analysis: Analysis) -> None:
    session.query(Finding).filter(Finding.analysis_id == analysis.id).delete()


def _apply_findings(session: Session, analysis: Analysis, findings: List[Dict]) -> None:
    for record in findings:
        session.add(
            Finding(
                analysis_id=analysis.id,
                finding_id=record["finding_id"],
                type=record["type"],
                tooth_label=record.get("tooth_label"),
                region=record.get("region", {}),
                severity=record.get("severity"),
                confidence=record.get("confidence"),
                model_key=record.get("model_key"),
                model_version=record.get("model_version"),
                extra=record.get("extra", {}),
                note=record.get("note"),
                confirmed=record.get("confirmed"),
            )
        )


def _mark_image_error(image: ImageStudy | None, session: Session) -> None:
    if image is not None:
        image.status = "error"
        session.add(image)


def process_analysis_job(analysis_id: str) -> None:
    """Background worker: run ML inference and persist results."""

    session: Session = SessionLocal()
    try:
        analysis = session.get(Analysis, analysis_id)
        if analysis is None:
            return

        image: ImageStudy | None = analysis.image
        if image is None or not image.storage_uri:
            analysis.status = "failed"
            analysis.completed_at = datetime.utcnow()
            analysis.overall_assessment = "找不到對應影像，無法完成分析。"
            session.add(analysis)
            _mark_image_error(image, session)
            session.commit()
            return

        model_config = session.scalars(select(ModelConfig).where(ModelConfig.is_active == True)).first()
        if model_config is None:
            analysis.status = "failed"
            analysis.completed_at = datetime.utcnow()
            analysis.overall_assessment = "尚未設定啟用的模型。"
            session.add(analysis)
            _mark_image_error(image, session)
            session.commit()
            return

        definition = ModelDefinition(
            model_id=model_config.id,
            detector_path=Path(model_config.detector_path),
            classifier_path=Path(model_config.classifier_path),
            detector_threshold=model_config.detector_threshold,
            classification_threshold=model_config.classification_threshold,
            max_teeth=model_config.max_teeth,
            updated_at=model_config.updated_at or datetime.utcnow(),
        )
        engine = get_inference_engine(definition)

        analysis.status = "in_progress"
        session.add(analysis)
        session.commit()

        image_path = Path(image.storage_uri)
        try:
            result = engine.analyze(image_path, analysis_id=analysis.id, image_id=image.id)
        except InferenceError as exc:
            analysis.status = "failed"
            analysis.completed_at = datetime.utcnow()
            analysis.overall_assessment = f"推論失敗：{exc}"
            session.add(analysis)
            _mark_image_error(image, session)
            session.commit()
            return

        _reset_findings(session, analysis)
        _apply_findings(session, analysis, result.get("findings", []))

        analysis.detected_conditions = result.get("detected_conditions", [])
        analysis.overall_assessment = result.get("overall_assessment")
        analysis.completed_at = datetime.utcnow()
        analysis.status = "completed"
        session.add(analysis)

        image.status = "pending_review"
        session.add(image)

        session.commit()
    except Exception as exc:  # pragma: no cover - 防止背景執行吞例外
        session.rollback()
        analysis = session.get(Analysis, analysis_id)
        if analysis is not None:
            analysis.status = "failed"
            analysis.completed_at = datetime.utcnow()
            analysis.overall_assessment = f"分析過程發生例外：{exc}"
            session.add(analysis)
            _mark_image_error(analysis.image, session)
            session.commit()
    finally:
        session.close()
