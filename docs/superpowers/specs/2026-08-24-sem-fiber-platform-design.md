# SEM Fiber Analysis Platform Design

## Goal

Turn the existing v6.11 Colab fiber-thickness model into a production-oriented web MVP where a user uploads an SEM image, runs asynchronous inference, reviews measurements in an interactive canvas, and approves the reviewed result into a persistent training dataset for later retraining.

## Scope

The MVP includes:

1. SEM image upload.
2. Asynchronous analysis jobs with `QUEUED`, `ANALYZING`, `POSTPROCESSING`, `DONE`, and `FAILED` states.
3. Python inference through the extracted `sem_fiber_ai` package from the v6.11 notebook, with the learned model as primary detector and opt-in wide-fiber recovery enabled by default for new-image analysis.
4. Next.js viewer with wheel zoom, drag pan, selectable measurement lines, endpoint editing, deletion, and manual line creation.
5. Immutable model predictions plus append-only review events.
6. Review completion that creates training examples classified as `AUTO_KEEP`, `AUTO_REMOVE`, `MANUAL_ADD`, or `MANUAL_CORRECT`.
7. Training snapshots that are accumulated but do not automatically retrain the production model.
8. A retraining export/CLI path that can later consume approved data together with the original answer-sheet dataset.
9. Docker Compose local stack using PostgreSQL, Redis/RQ, MinIO, FastAPI, worker, and Next.js.

Out of scope for this MVP: authentication, multi-tenant projects, automated champion promotion, batch analysis UI, admin UI, and automatic retraining on every review.

## Architecture

`Next.js -> FastAPI -> PostgreSQL + S3-compatible object storage`, with inference dispatched to an RQ worker through Redis. The worker calls a narrow `InferenceEngine` interface; its production implementation wraps the v6.11 `sem_fiber_ai.src.infer` package. The API never imports the heavy model package while serving ordinary HTTP requests.

Model output is immutable. Review edits create/update a review working set and append `ReviewEvent` records. Completing a review freezes the final geometry and materializes `TrainingExample` rows that preserve both the original model proposal and the reviewer decision.

## Data model

- `ImageAsset`: uploaded SEM metadata and object-storage key.
- `AnalysisJob`: state, progress, error, model version, timestamps.
- `ModelMeasurement`: immutable inference result (`x1/y1/x2/y2`, width, angle, confidence, source).
- `ReviewSession`: one review per completed analysis, state `OPEN` or `APPROVED`.
- `ReviewMeasurement`: mutable review working geometry linked to an optional source model measurement.
- `ReviewEvent`: append-only `KEEP`, `REMOVE`, `ADD`, `CORRECT` audit event.
- `TrainingExample`: frozen supervision generated only at approval time.
- `ModelVersion`: checkpoint identifier and deployment metadata.

## Review semantics

- Untouched active model measurement at approval -> `AUTO_KEEP`.
- Removed model measurement -> `AUTO_REMOVE`.
- Manual line with no source model measurement -> `MANUAL_ADD`.
- Edited model measurement -> `MANUAL_CORRECT`, storing both original and corrected geometry.

`AUTO_REMOVE` means “the reviewer rejected measurement at this site,” not necessarily “this pixel is not fiber.” Training export therefore exposes a separate `measure_here` target and does not globally label all unmarked pixels as negatives.

## Inference contract

`InferenceEngine.analyze(image_path, output_dir, nm_per_pixel=None) -> AnalysisResult`

Each returned measurement contains:

- `x1`, `y1`, `x2`, `y2`
- `width_px`, optional `width_nm`
- `angle_deg`
- `confidence`
- `source` (`ai`, `thick_recovery`, or adapter-specific source)
- optional `validity`, `uncertainty`, and diagnostic metadata

The adapter loads the checkpoint once per worker process. Checkpoint and optional width-calibration JSON are mounted through configuration, not committed to Git.

## Training lifecycle

Approval adds immutable supervision to the database. A separate export command writes an approved dataset bundle (JSONL + image references/objects) and can be followed by a retraining command. New checkpoints are candidates only; automatic deployment is deliberately excluded until a fixed validation/holdout gate is implemented.

## Error handling

- Unsupported upload -> HTTP 415/422.
- Missing analysis checkpoint -> job becomes `FAILED` with a clear worker error.
- Inference exception -> transaction marks job `FAILED`; partial model measurements are not committed.
- Editing a non-open review -> HTTP 409.
- Approving twice -> idempotently returns the already-approved review/training count.
- Storage failures -> request/job fails without creating misleading `DONE` state.

## Testing

- Backend unit tests use SQLite and an in-memory fake object store/queue where appropriate.
- Inference adapter mapping is tested with a fake `sem_fiber_ai` DataFrame rather than requiring a model checkpoint.
- Review tests verify all four supervision classes and idempotent approval.
- API tests verify upload, job creation, result retrieval, editing, and approval.
- Frontend tests focus on geometry reducer behavior; TypeScript build/typecheck verifies the canvas component integration.
- Extracted `sem_fiber_ai` package retains its original tests and is smoke-tested independently where dependencies are available.
