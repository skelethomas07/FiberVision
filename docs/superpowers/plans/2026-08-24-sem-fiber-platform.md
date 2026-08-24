# SEM Fiber Analysis Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a runnable GitHub-ready MVP that serves the v6.11 SEM fiber model through asynchronous FastAPI jobs, lets reviewers edit measurements in Next.js, and turns approved reviews into future training examples.

**Architecture:** FastAPI owns HTTP, PostgreSQL persistence, and object-storage metadata; RQ workers own heavy inference and import the extracted v6.11 `sem_fiber_ai` package. Next.js renders/edit lines while the backend preserves immutable model predictions and freezes approved review decisions into training examples.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2, PostgreSQL, Redis/RQ, boto3/MinIO, PyTorch/scikit-image model package, Next.js 15-16 compatible, React 19, TypeScript, Vitest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-24-sem-fiber-platform-design.md`

## Global Constraints

- Colab is not a production backend.
- Model predictions remain immutable after inference.
- Approved review decisions are persisted as `AUTO_KEEP`, `AUTO_REMOVE`, `MANUAL_ADD`, or `MANUAL_CORRECT`.
- Approval adds data to the training dataset but does not automatically retrain or deploy a model.
- The v6.11 extracted package is the production inference implementation; checkpoint files are external configuration and are never committed.
- Wide-fiber recovery is provenance-labelled separately from learned AI measurements.

---

### Task 1: Repository and model package boundary

**Files:**
- Create: `backend/app/inference/contracts.py`
- Create: `backend/app/inference/sem_fiber_engine.py`
- Create: `backend/vendor/sem_fiber_ai/**`
- Test: `backend/tests/test_inference_adapter.py`

**Interfaces:**
- Produces: `InferenceEngine.analyze(Path, Path, float | None) -> AnalysisResult`.
- Produces: canonical `MeasurementPrediction` objects consumed by the worker.

- [x] Write a failing adapter test using a fake inference DataFrame with `x1_px/y1_px/x2_px/y2_px`, `width_px`, `width_nm`, `confidence`, and `measurement_source`.
- [x] Run the adapter test and verify it fails before the contract exists.
- [x] Add the contract dataclasses and mapper; vendor the extracted v6.11 package without checkpoint weights.
- [x] Implement `SemFiberEngine` with lazy checkpoint loading and default thick recovery.
- [x] Run the adapter test and vendored lightweight tests.
- [x] Commit the task.

### Task 2: Persistence, storage, and async jobs

**Files:**
- Create: `backend/app/config.py`, `backend/app/db.py`, `backend/app/models.py`, `backend/app/schemas.py`
- Create: `backend/app/storage.py`, `backend/app/queue.py`, `backend/app/services/analysis.py`, `backend/app/workers/analysis.py`
- Test: `backend/tests/test_analysis_service.py`

**Interfaces:**
- Consumes: `InferenceEngine.analyze`.
- Produces: `create_analysis`, `run_analysis_job`, `get_analysis_result`.

- [x] Write failing tests for job state transitions and atomic measurement persistence.
- [x] Run them and confirm failure.
- [x] Implement SQLAlchemy entities, local/S3-compatible storage abstraction, RQ enqueue abstraction, and analysis service.
- [x] Implement worker state transitions `QUEUED -> ANALYZING -> POSTPROCESSING -> DONE` and `FAILED` handling.
- [x] Run tests and commit.

### Task 3: Review and training supervision

**Files:**
- Create: `backend/app/services/review.py`
- Create: `backend/app/training/export.py`
- Test: `backend/tests/test_review_training.py`

**Interfaces:**
- Produces: `get_or_create_review`, `apply_review_changes`, `approve_review`, `export_approved_dataset`.

- [x] Write failing tests covering untouched model prediction -> `AUTO_KEEP`, deletion -> `AUTO_REMOVE`, manual addition -> `MANUAL_ADD`, endpoint edit -> `MANUAL_CORRECT`, and repeated approval idempotency.
- [x] Run tests and confirm failure.
- [x] Implement working review measurements and append-only events.
- [x] Implement approval transaction that freezes `TrainingExample` rows including original/corrected geometry.
- [x] Implement JSONL dataset export.
- [x] Run tests and commit.

### Task 4: FastAPI surface

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/app/api/images.py`, `backend/app/api/analyses.py`, `backend/app/api/reviews.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Produces endpoints: `POST /api/images`, `POST /api/analyses`, `GET /api/analyses/{id}`, `GET /api/analyses/{id}/result`, `GET /api/analyses/{id}/review`, `PATCH /api/reviews/{id}`, `POST /api/reviews/{id}/approve`.

- [x] Write API tests around upload -> analysis -> result -> review -> approval.
- [x] Run tests and confirm failure.
- [x] Implement routers and validation with consistent API schemas.
- [x] Run tests and commit.

### Task 5: Next.js upload/status/result viewer

**Files:**
- Create: `frontend/app/page.tsx`, `frontend/app/analysis/[id]/page.tsx`
- Create: `frontend/components/MeasurementCanvas.tsx`, `frontend/components/AnalysisStatus.tsx`, `frontend/lib/api.ts`, `frontend/lib/geometry.ts`
- Test: `frontend/lib/geometry.test.ts`

**Interfaces:**
- Consumes FastAPI endpoints from Task 4.
- Produces an editable canvas payload `{removedIds, added, corrected}`.

- [x] Write reducer/geometry tests for line width/angle and edit payload generation.
- [x] Implement upload and analysis creation.
- [x] Implement polling status screen.
- [x] Implement canvas wheel zoom, drag pan, selection, endpoint dragging, deletion, and manual line creation.
- [x] Implement save and approve actions.
- [x] Run frontend tests/typecheck/build and commit.

### Task 6: Docker, retraining entrypoint, documentation, verification

**Files:**
- Create: `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile`, `.env.example`
- Create: `backend/app/training/cli.py`
- Create/Modify: `README.md`

**Interfaces:**
- Produces local production-like stack and commands `python -m app.training.cli export` and `python -m app.training.cli retrain`.

- [x] Add Docker services for Postgres, Redis, MinIO, API, worker, and frontend.
- [x] Add training CLI that exports approved snapshots and invokes vendored `sem_fiber_ai.src.train` with an explicit config when retraining is requested.
- [x] Document checkpoint mounting, object storage, local start, review lifecycle, and retraining lifecycle.
- [x] Run backend tests, frontend tests/typecheck/build, compile checks, and Docker Compose config validation when Docker is available.
- [x] Record verification output in `VERIFICATION.md` and commit.
