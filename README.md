# SEM Fiber Analysis Platform

GitHub-ready MVP for automatic SEM fiber thickness analysis with human review and continual dataset growth.

## What this repository does

1. Upload an SEM image in the Next.js UI.
2. FastAPI creates an asynchronous analysis job.
3. An RQ worker loads the trained SEM model and runs inference outside the web process.
4. The v6.11 notebook model is the primary detector; its wide-fiber recovery branch is enabled for new-image analysis and keeps its provenance (`ai` vs `thick_recovery`).
5. The browser displays editable measurement lines with wheel zoom and drag pan.
6. A reviewer can keep, remove, add, or correct lines.
7. `검수 완료` freezes the review into training supervision.
8. A later retraining run can merge those approved examples with the original answer-sheet dataset.

Colab is not used as a production backend.

## Important model provenance

The supplied final notebook `sem_fiber_ai_colab_v6_11` embeds a complete Python package. That package was extracted into `backend/vendor/sem_fiber_ai`. Its internal `__version__` is `6.6.0`; this repository calls the deployed experiment/model line `v6.11` because that is the supplied notebook revision. No trained checkpoint is committed.

Place your trained checkpoint at:

```text
models/model.pt
```

or change `MODEL_CHECKPOINT`.

The older VisionFlux/Streamlit and `chem_frontier_learning.py` files are retained under `legacy/` for reference. The production stack does not import them.

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
       +-- sem_fiber_ai learned detector
       +-- wide-fiber recovery
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
mkdir -p models
cp /path/to/your/checkpoint.pt models/model.pt
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
POST /api/images
POST /api/analyses
GET  /api/analyses/{id}
GET  /api/analyses/{id}/result
GET  /api/analyses/{id}/review
PATCH /api/reviews/{review_id}
POST /api/reviews/{review_id}/approve
```

Job states:

```text
QUEUED -> ANALYZING -> POSTPROCESSING -> DONE
                                   \-> FAILED
```

## Add reviewed images to the next training dataset

Approving a review immediately creates immutable `training_examples` rows in PostgreSQL. It does **not** automatically retrain or replace the deployed model.

Export audit-friendly JSONL:

```bash
cd backend
PYTHONPATH=. python -m app.training.cli export \
  --output training_exports/approved.jsonl
```

Materialize data in the format consumed by the vendored v6.11 training package:

```bash
PYTHONPATH=. python -m app.training.cli prepare \
  --output-dir training_exports/reviewed_bundle
```

This creates:

```text
reviewed_bundle/
├── labels.csv
└── images/
```

`AUTO_REMOVE` rows become `is_negative=True` reviewer-rejected sites. Positive rows contain `center_x_px`, endpoints, width, measurement angle, derived local fiber angle, nm/px, and the original supervision label.

To merge the reviewed examples with the original answer-sheet training data and launch a candidate retrain:

```bash
PYTHONPATH=. python -m app.training.cli retrain \
  --base-labels-csv /data/original_processed/labels.csv \
  --base-image-dir /data/original_images \
  --dataset-dir training_exports/retrain_dataset \
  --output-dir training_runs/candidate_v2 \
  --init-from /models/model.pt
```

The command writes the exact generated training YAML into the candidate run directory and then calls `sem_fiber_ai.src.train.train()`.

**Deployment is intentionally manual.** A new candidate should first be compared on a fixed image-level holdout (width distribution, sampling density, fiber-level recall, and any precision floor you decide to enforce) before changing `MODEL_CHECKPOINT`.

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

- No authentication or per-user project isolation yet.
- No automatic champion promotion.
- No batch UI.
- Database schema is created with SQLAlchemy `create_all`; add Alembic before a long-lived production deployment with schema migrations.
- GPU deployment may need a CUDA-specific PyTorch base image instead of the default CPU-friendly Python image.
