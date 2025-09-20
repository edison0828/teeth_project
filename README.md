# Oral X-Ray Intelligence Platform

This repository combines interactive web tooling with existing research code to accelerate oral radiograph analysis. The new implementation follows the architecture proposed in `docs/oral_xray_system_design.md`, delivering a cohesive FastAPI backend and a Next.js App Router frontend inspired by the provided UI references.

## Repository layout

```
teeth_project/
├── backend/                  # FastAPI application with in-memory sample data
│   ├── main.py                # REST endpoints, orchestration helpers
│   ├── schemas.py             # Pydantic domain models reflecting the design doc
│   └── requirements.txt       # Minimal API dependencies
├── frontend/                 # Next.js 14 (App Router) interface styled after the UI mocks
│   ├── app/                   # Dashboard, Patients, Upload, Analysis result pages
│   ├── components/            # Shared navigation, cards, progress widgets
│   ├── lib/                   # Typed API client with graceful fallbacks
│   └── package.json           # Web dependencies and scripts
├── docs/oral_xray_system_design.md
├── src/                      # Legacy model training/inference scripts (PyTorch)
└── ...
```

## Getting started

### 1. Backend API service

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

The development server exposes:

- `GET /api/dashboard/overview` – consolidated statistics, queue insights, recent patients.
- `GET /api/patients` / `POST /api/patients` – patient registry operations.
- `GET /api/analyses/{id}` – standardized analysis details, findings, and timeline metadata.
- `POST /api/images` / `POST /api/analyses` – simulate upload + AI orchestration.

Responses currently use in-memory sample data so the UI works immediately while real integrations are in progress.

### 2. Frontend workspace

```bash
cd frontend
npm install
npm run dev
```

By default the client talks to `http://localhost:8000`. To point to another instance, set `NEXT_PUBLIC_API_BASE_URL` before running the dev server.

### 3. Preview

- Dashboard: quick actions, statistics gauge, condition distribution, and live queue cards.
- Patient management: sortable roster with a contextual detail panel (demographics, history, recent analyses).
- Image upload: drag-and-drop dropzone, preprocessing presets, patient/study bindings.
- Analysis result: radiograph canvas placeholder, findings list, detected-condition summary, pipeline timeline, and export actions.

All pages degrade gracefully to curated mock data if the API is offline, enabling UI work even without the backend.

## Legacy deep-learning utilities

The original PyTorch training scripts (`src/`), dataset utilities, and accompanying documentation remain unchanged for researchers who need to continue model experimentation. Refer to the comments inside each script and existing CSV/DICOM preparation notes when running those workflows.

## Next steps

- Replace in-memory stores with persistent services (SQL + object storage) and background job orchestration.
- Wire the upload form to the real `/api/images` endpoint with file handling/presigned URLs.
- Expand authentication/authorization according to the design proposal (OAuth2 + RBAC).
