"""Database layer and ORM models for the Oral X-Ray backend."""
from __future__ import annotations

import os
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy import inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "teeth.db"
    DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Base class for declarative SQLAlchemy models."""


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(120))
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sessions: Mapped[list["UserSession"]] = relationship(
        "UserSession", back_populates="user", cascade="all, delete-orphan"
    )


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.utcnow() + timedelta(days=7))
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    user_agent: Mapped[Optional[str]] = mapped_column(String(255))

    user: Mapped[User] = relationship("User", back_populates="sessions")


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    dob: Mapped[date] = mapped_column(Date, nullable=False)
    gender: Mapped[str] = mapped_column(String(30), nullable=False)
    contact: Mapped[Optional[str]] = mapped_column(String(150))
    medical_history: Mapped[Optional[str]] = mapped_column(Text)
    last_visit: Mapped[Optional[date]] = mapped_column(Date)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    images: Mapped[list["ImageStudy"]] = relationship(
        "ImageStudy", back_populates="patient", cascade="all, delete-orphan"
    )


class ImageStudy(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(50), default="uploaded")
    storage_uri: Mapped[Optional[str]] = mapped_column(String(255))
    original_filename: Mapped[Optional[str]] = mapped_column(String(255))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    auto_analyze: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    patient: Mapped["Patient"] = relationship("Patient", back_populates="images")
    analyses: Mapped[list["Analysis"]] = relationship(
        "Analysis", back_populates="image", cascade="all, delete-orphan"
    )


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), nullable=False, index=True)
    requested_by: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    priority: Mapped[str] = mapped_column(String(20), default="standard")
    triggered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    overall_assessment: Mapped[Optional[str]] = mapped_column(Text)
    detected_conditions: Mapped[list[dict]] = mapped_column(JSON, default=list)

    image: Mapped["ImageStudy"] = relationship("ImageStudy", back_populates="analyses")
    findings: Mapped[list["Finding"]] = relationship(
        "Finding", back_populates="analysis", cascade="all, delete-orphan"
    )


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), nullable=False, index=True)
    finding_id: Mapped[str] = mapped_column(String(40), nullable=False)
    type: Mapped[str] = mapped_column(String(40), nullable=False)
    tooth_label: Mapped[Optional[str]] = mapped_column(String(40))
    region: Mapped[dict] = mapped_column(JSON, default=dict)
    severity: Mapped[Optional[str]] = mapped_column(String(20))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    model_key: Mapped[Optional[str]] = mapped_column(String(60))
    model_version: Mapped[Optional[str]] = mapped_column(String(40))
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    note: Mapped[Optional[str]] = mapped_column(Text)
    confirmed: Mapped[Optional[bool]] = mapped_column(Boolean)

    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="findings")



class ModelConfig(Base):
    __tablename__ = "model_configs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    model_type: Mapped[str] = mapped_column(String(40), default="cross_attn")
    detector_path: Mapped[str] = mapped_column(String(255), nullable=False)
    classifier_path: Mapped[str] = mapped_column(String(255), nullable=False)
    detector_threshold: Mapped[float] = mapped_column(Float, default=0.25)
    classification_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    max_teeth: Mapped[int] = mapped_column(Integer, default=64)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def _ensure_model_config_schema(connection) -> None:
    """Ensure the ``model_configs`` table contains expected columns."""

    inspector = inspect(connection)
    if "model_configs" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("model_configs")}
    if "model_type" not in existing_columns:
        connection.execute(
            text("ALTER TABLE model_configs ADD COLUMN model_type VARCHAR(40) DEFAULT 'cross_attn'")
        )
        connection.execute(
            text("UPDATE model_configs SET model_type = 'cross_attn' WHERE model_type IS NULL")
        )


def init_db() -> None:
    """Create database tables if they do not already exist and run lightweight migrations."""

    Base.metadata.create_all(bind=engine)

    with engine.begin() as connection:
        _ensure_model_config_schema(connection)


def get_session() -> Generator[Session, None, None]:
    """Return a scoped SQLAlchemy session for dependency injection."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
