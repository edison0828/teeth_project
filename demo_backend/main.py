from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import DemoSettings
from .pipeline import CrossAttentionDemoPipeline, DemoInferenceError
from .samples import SampleStore
from .schemas import (
    DemoError,
    DemoInferenceFinding,
    DemoInferenceResponse,
    DemoSampleListResponse,
)

settings = DemoSettings()
settings.ensure_directories()

pipeline = CrossAttentionDemoPipeline(settings)
sample_store = SampleStore(settings)

app = FastAPI(
    title="Cross Attention Demo API",
    description="Minimal FastAPI service that exposes the cross-attention Grad-CAM demo endpoints.",
    version="0.1.0",
    openapi_url="/demo/openapi.json",
    docs_url="/demo/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.static_dir.exists():
    app.mount("/demo-assets", StaticFiles(directory=settings.static_dir), name="demo-assets")
app.mount("/demo-outputs", StaticFiles(directory=settings.output_dir), name="demo-outputs")


def get_pipeline() -> CrossAttentionDemoPipeline:
    return pipeline


def get_sample_store() -> SampleStore:
    return sample_store


@app.on_event("startup")
async def _startup() -> None:
    sample_store.refresh()
    if settings.autoload_model:
        await run_in_threadpool(pipeline.ensure_loaded)


@app.get("/demo/health", response_model=dict)
async def health() -> dict:
    return {"status": "ok"}


@app.get(
    "/demo/samples",
    response_model=DemoSampleListResponse,
    responses={200: {"description": "List of curated demo samples."}},
)
async def list_samples(store: SampleStore = Depends(get_sample_store)) -> DemoSampleListResponse:
    return DemoSampleListResponse(items=store.to_response())


def _resolve_static_path(uri: str) -> Path:
    if uri.startswith("/demo-assets/"):
        relative = uri[len("/demo-assets/") :]
        return settings.static_dir / relative
    return Path(uri)


async def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.png").suffix
    temp_path = settings.output_dir / f"upload-{uuid4().hex}{suffix}"
    contents = await file.read()
    temp_path.write_bytes(contents)
    return temp_path


@app.post(
    "/demo/infer",
    response_model=DemoInferenceResponse,
    responses={
        400: {"model": DemoError},
        404: {"model": DemoError},
        422: {"model": DemoError},
    },
)
async def run_inference(
    sample_id: Optional[str] = Form(default=None),
    rerun: bool = Form(default=False),
    file: Optional[UploadFile] = File(default=None),
    pipeline: CrossAttentionDemoPipeline = Depends(get_pipeline),
    store: SampleStore = Depends(get_sample_store),
) -> DemoInferenceResponse:
    if file is None and sample_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide a sample_id or upload an image.")

    # Pre-computed sample replay path
    if sample_id and not rerun and file is None:
        sample = store.get(sample_id)
        if sample is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample not found.")

        findings = [
            DemoInferenceFinding(**finding, cam_path=sample.cam_paths.get(finding["fdi"]))
            for finding in sample.findings
        ]
        overlay = sample.overlay_path or sample.image_path
        return DemoInferenceResponse(
            request_id=f"sample-{sample.sample_id}",
            overlay_url=overlay,
            csv_url="",
            findings=findings,
            warnings=["Returning pre-computed assets bundled with the demo."],
        )

    image_path: Optional[Path] = None
    cleanup_path: Optional[Path] = None

    if file is not None:
        image_path = await _save_upload(file)
        cleanup_path = image_path
    elif sample_id:
        sample = store.get(sample_id)
        if sample is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample not found.")
        image_path = _resolve_static_path(sample.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample image missing on server.")

    assert image_path is not None

    try:
        prediction = await run_in_threadpool(pipeline.predict, image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except DemoInferenceError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    finally:
        if cleanup_path and cleanup_path.exists():
            cleanup_path.unlink(missing_ok=True)

    findings = [
        DemoInferenceFinding(**finding, cam_path=None)
        for finding in prediction.findings
    ]

    return DemoInferenceResponse(
        request_id=prediction.request_id,
        overlay_url=prediction.overlay_url(settings),
        csv_url=prediction.csv_url(settings),
        findings=findings,
        warnings=prediction.warnings,
    )
