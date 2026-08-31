# SEM Fiber Analysis Platform

GitHub-ready MVP for automatic SEM fiber thickness analysis with human review and continual dataset growth.

## What this repository does

1. Upload an SEM image in the Next.js UI.
2. FastAPI creates an asynchronous analysis job.
3. An RQ worker loads the trained SEM model and runs inference outside the web process.
4. The embedded `sem_fiber_ai` v7.0.0 package loads the deployed `best.pt` checkpoint and produces geometry-based thickness measurements with per-site confidence.
5. The browser displays editable measurement lines with wheel zoom and drag pan.
6. A reviewer can keep, remove, add, or correct lines.
7. `검수 완료` freezes the review into training supervision.
8. A later retraining run can merge those approved examples with the original answer-sheet dataset.

Colab is not used as a production backend.

## Important model provenance

Production inference uses the `sem_fiber_ai 7.0.0` package embedded in the supplied v7 notebook. The package is vendored under `backend/vendor/sem_fiber_ai`; the trained checkpoint stays outside Git and is mounted into the worker.

Place the v7 run files at:

```text
models/v7/best.pt
models/v7/physical_reference.json
```

`selection.json` and `training_stats.json` are optional. When `selection.json` is absent, v7 uses the post-processing defaults stored in the checkpoint config.

The older VisionFlux/Streamlit and historical learning files remain under `legacy/` for reference only. The production stack does not import them.

## Architecture

```text
Browser / Next.js
       |
       v
     FastAPI  ---- PostgreSQL
       |       ---- MinIO / S3-compatible storage
       |
       v
     Redis / RQ
       |
       v
 Python inference worker
       |
       +-- sem_fiber_ai v7 geometry detector
```

The API process does not load PyTorch. The RQ worker imports the model lazily and keeps one engine instance in the worker process.

## Review -> training semantics

Model predictions are immutable. The review layer is separate.

| Reviewer action | Training label | `is_fiber` | `measure_here` |
|---|---|---:|---:|
| Keep model line | `AUTO_KEEP` | true | true |
| Remove model line | `AUTO_REMOVE` | unknown | false |
| Add new line | `MANUAL_ADD` | true | true |
| Move/resize model line | `MANUAL_CORRECT` | true | true |

`AUTO_REMOVE` is deliberately **not** treated as “definitely not a fiber.” It means the reviewer rejected measurement at that site. This preserves the intended separation between `P(is_fiber)` and `P(measure_here)`.

## Local start with Docker

Prerequisites: Docker + Docker Compose and the trained checkpoint.

```bash
cp .env.example .env
mkdir -p models/v7
cp /path/to/your/best.pt models/v7/best.pt
cp /path/to/your/physical_reference.json models/v7/physical_reference.json
docker compose up --build
```

Open:

- Web: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`
- MinIO console: `http://localhost:9001`

Change the sample MinIO password before exposing this outside a local network.

## Development without Docker

Backend API tests do not require a checkpoint because the model adapter is lazy and worker tests use a fake engine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-api.txt
PYTHONPATH=backend pytest -q backend/tests
```

For full inference install model dependencies too:

```bash
pip install -r backend/requirements.txt
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## HTTP flow

```text
POST /api/auth/login
GET  /api/auth/me
POST /api/auth/change-password
POST /api/auth/logout
POST /api/images
POST /api/analyses
GET  /api/analyses/{id}
GET  /api/analyses/{id}/result
GET  /api/analyses/{id}/review
PATCH /api/reviews/{review_id}
POST /api/reviews/{review_id}/approve
GET  /api/analyses/{id}/exports/csv
GET  /api/analyses/{id}/exports/overlay
GET  /api/analyses/{id}/exports/labeled
GET  /api/analyses/{id}/exports/bundle
```

Job states:

```text
QUEUED -> ANALYZING -> POSTPROCESSING -> DONE
                                   \-> FAILED
```

## Access control and user provisioning

FiberVision has no public signup page. Accounts are created by an administrator inside the API container:

```bash
docker compose exec api python -m app.cli.create_user user@example.com
```

The command securely prompts for the initial password twice. On the user's first login, FiberVision requires a new password before image, analysis, review, or export APIs can be used. Passwords are stored as Argon2id hashes and browser sessions use an HttpOnly cookie.

Current EC2 deployment uses HTTP on a restricted security-group source IP, so `AUTH_COOKIE_SECURE=false`. Use a unique FiberVision password while HTTP is in use; set the value to `true` after HTTPS is enabled.

Authentication settings:

```text
AUTH_SESSION_DAYS=7
AUTH_COOKIE_NAME=fibervision_session
AUTH_COOKIE_SECURE=false
```

Analysis results expose three work areas: `검수`, `결과`, and `내보내기`. Export downloads include CSV, final measurement overlay PNG, numbered-label PNG, and a ZIP bundle.

## Add reviewed images to the next training dataset

Approving a review immediately creates immutable `training_examples` rows in PostgreSQL. It does **not** automatically retrain or replace the deployed v7 model.

Export audit-friendly JSONL:

```bash
cd backend
PYTHONPATH=. python -m app.training.cli export \
  --output training_exports/approved.jsonl
```

Or materialize the approved review measurements and source images for a future v7 training-data integration:

```bash
PYTHONPATH=. python -m app.training.cli prepare \
  --output-dir training_exports/reviewed_bundle
```

The old v6 retrain command has been removed. The current v7 checkpoint was trained with the supplied v7 notebook protocol; a future retraining workflow should explicitly adapt approved FiberVision supervision to that protocol before replacing `best.pt`.

## Repository layout

```text
backend/
  app/                    FastAPI, DB, review/training services, worker
  vendor/sem_fiber_ai/    model package extracted from final notebook
  tests/
frontend/
  app/                    upload + analysis pages
  components/             editable canvas
legacy/                   supplied VisionFlux/reference code
models/                   checkpoint mount point (weights gitignored)
docs/superpowers/         design + implementation plan
```

## Current MVP limits

- Authentication is enabled; analysis records are not yet isolated into per-user project/history views.
- No automatic champion promotion.
- No batch UI.
- Database schema is created with SQLAlchemy `create_all`; add Alembic before a long-lived production deployment with schema migrations.
- GPU deployment may need a CUDA-specific PyTorch base image instead of the default CPU-friendly Python image.
