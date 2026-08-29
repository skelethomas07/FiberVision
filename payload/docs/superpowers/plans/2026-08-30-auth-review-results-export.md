# FiberVision Auth, Review Results, and Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add administrator-provisioned authentication, a cleaner VisionFlux-style review workspace, 6/9/12/16 sector fitting, result charts, and CSV/PNG/ZIP exports without breaking existing inference or training flows.

**Architecture:** FastAPI owns authentication/session validation and server-generated exports using the existing PostgreSQL and object storage. Next.js remains a thin client with a protected login flow and a three-tab analysis workspace; pure TypeScript helpers compute sector geometry and chart data. Existing analysis/review tables are preserved and only new authentication tables are created.

**Tech Stack:** FastAPI, SQLAlchemy, Argon2id (`argon2-cffi`), PostgreSQL, Pillow, Next.js 15/16-compatible React 19, TypeScript, native Canvas/SVG, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-30-auth-review-results-export-design.md`

## Global Constraints

- No public signup or web admin screen.
- CLI account creation must prompt for the initial password instead of accepting it in shell history.
- Password hashes use Argon2id; raw passwords and raw session tokens are never persisted.
- Session cookie is HttpOnly + SameSite=Lax; Secure is controlled by configuration because current production uses HTTP.
- Existing v6.11 inference, VisionFlux import, TIFF preview, review labels, approval, and training-example generation remain intact.
- Split modes are exactly 6, 9, 12, and 16 with 3×2, 3×3, 4×3, and 4×4 grids.
- Low-confidence filter threshold is 0.70.
- Exports contain saved active review measurements only.
- UI navigation labels are `검수`, `결과`, and `내보내기`; review tool labels remain short and uncluttered.

---

### Task 1: Authentication domain and service

**Files:**
- Modify: `backend/app/models.py`
- Modify: `backend/app/config.py`
- Modify: `backend/requirements-api.txt`
- Create: `backend/app/services/auth.py`
- Create: `backend/tests/test_auth_service.py`

**Interfaces:**
- Produces: `create_user(session, email, password)`, `authenticate_user(session, email, password)`, `create_session(session, user)`, `resolve_session(session, token)`, `change_password(session, user, password)`, `revoke_session(session, token)`.

- [ ] Write service tests for normalized email, Argon2 verification, initial-password flag, opaque session resolution, expiration, password change, and logout invalidation.
- [ ] Run `PYTHONPATH=backend pytest -q backend/tests/test_auth_service.py` and verify the new tests fail because auth models/service do not exist.
- [ ] Add `User` and `AuthSession` tables plus auth settings and Argon2 dependency.
- [ ] Implement the minimal auth service satisfying the tests.
- [ ] Re-run the auth service tests and the existing backend suite.

### Task 2: Auth HTTP API and protected application routes

**Files:**
- Create: `backend/app/api/auth.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_api.py`
- Create: `backend/tests/test_auth_api.py`

**Interfaces:**
- Produces endpoints: `POST /api/auth/login`, `GET /api/auth/me`, `POST /api/auth/change-password`, `POST /api/auth/logout`.
- Produces dependency: `require_user(request)` and `require_ready_user(request)`.

- [ ] Write API tests proving anonymous analysis access is 401, valid login sets a cookie, first-login users are blocked from analysis until password change, and logout invalidates access.
- [ ] Run the targeted tests and verify the expected 401/404/import failures.
- [ ] Implement auth router and router-level authentication dependencies while keeping `/healthz` public.
- [ ] Update the existing end-to-end API test to create/login a user before upload.
- [ ] Run all backend tests.

### Task 3: Secure administrator CLI account provisioning

**Files:**
- Create: `backend/app/cli/__init__.py`
- Create: `backend/app/cli/create_user.py`
- Create: `backend/tests/test_create_user_cli.py`

**Interfaces:**
- Command: `python -m app.cli.create_user user@example.com`; prompts twice with `getpass`, creates the account with `must_change_password=true`, and rejects duplicate email.

- [ ] Write CLI unit tests around the callable `main(argv=None, password_reader=getpass.getpass)`.
- [ ] Verify tests fail before implementation.
- [ ] Implement CLI with normalized email and secure password prompt.
- [ ] Verify CLI tests and full backend suite pass.

### Task 4: Export service and download API

**Files:**
- Create: `backend/app/services/exports.py`
- Create: `backend/app/api/exports.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_exports.py`

**Interfaces:**
- Produces: `build_csv(rows) -> bytes`, `render_overlay(image_bytes, rows, labeled=False) -> bytes`, `build_export_zip(...) -> bytes`.
- Endpoints: `/api/analyses/{analysis_id}/exports/csv`, `/overlay`, `/labeled`, `/bundle`.

- [ ] Write failing tests for CSV ordering/content, active-only filtering, PNG dimensions/colors, labels, ZIP members, auth protection, and content-disposition headers.
- [ ] Run targeted export tests and verify failure.
- [ ] Implement pure export functions with Pillow and `csv`/`zipfile` from the standard library.
- [ ] Add protected download endpoints loading the browser-safe SEM preview and saved review rows.
- [ ] Run targeted tests then all backend tests.

### Task 5: Review payload confidence and frontend pure helpers

**Files:**
- Modify: `backend/app/api/reviews.py`
- Modify: `frontend/lib/geometry.ts`
- Modify: `frontend/lib/geometry.test.ts`
- Create: `frontend/lib/reviewView.ts`
- Create: `frontend/lib/reviewView.test.ts`

**Interfaces:**
- Review API adds `confidence: number | null` without changing persisted review schema.
- Produces `sectorGrid`, `sectorBounds`, `normalizeFiberAngle`, `measurementStats`, `histogramBins`, and `directionBins`.

- [ ] Write failing backend assertion that model-derived review rows expose confidence while manual rows expose null.
- [ ] Write failing Node tests for all four sector layouts, bounds, 0–180 normalization, summary statistics, histogram bins, and direction bins.
- [ ] Run targeted backend and frontend tests and verify failure.
- [ ] Implement backend confidence enrichment and pure frontend helpers.
- [ ] Re-run targeted tests.

### Task 6: Frontend authentication flow

**Files:**
- Modify: `frontend/lib/api.ts`
- Create: `frontend/components/AuthGuard.tsx`
- Create: `frontend/app/login/page.tsx`
- Create: `frontend/app/change-password/page.tsx`
- Modify: `frontend/app/layout.tsx`

**Interfaces:**
- API client always uses `credentials: "include"` and exposes `login`, `me`, `changePassword`, `logout`.
- `AuthGuard` redirects anonymous users to `/login` and first-login users to `/change-password`.

- [ ] Add a pure API error helper test if necessary so 401/403 states can be identified reliably.
- [ ] Implement credentialed API calls and auth types.
- [ ] Build login/change-password pages and reusable auth guard.
- [ ] Run frontend tests and TypeScript typecheck.

### Task 7: Clean home/upload screen

**Files:**
- Modify: `frontend/app/page.tsx`
- Modify: `frontend/app/globals.css`

**Interfaces:**
- Protected home with a single drag/drop upload surface, optional nm/pixel, account/logout action, and minimal VisionFlux/raw-SEM hint.

- [ ] Implement home inside `AuthGuard`, preserving current upload/create-analysis behavior.
- [ ] Add drag/drop and selected-file visual states without adding dependencies.
- [ ] Apply neutral professional analysis-tool styling and responsive rules.
- [ ] Run frontend typecheck/build.

### Task 8: VisionFlux-style review canvas and sector navigation

**Files:**
- Modify: `frontend/components/MeasurementCanvas.tsx`
- Modify: `frontend/app/globals.css`

**Interfaces:**
- Tool modes: select, add, edit, delete.
- Sector modes: whole, 6, 9, 12, 16; `fitCurrentSector()` fits the current sector.
- Provides undo/redo, overlay visibility, low-confidence-only filter, magnifier, selected inspector.

- [ ] Wire sector geometry helpers and verify 6/9/12/16 controls calculate current sector fit from image/world coordinates.
- [ ] Add snapshot-based undo/redo only at completed mutations (add, delete, endpoint correction), not on pointer-move frames.
- [ ] Add overlay/filter controls and keep manual lines visible outside low-confidence model filtering only when all view is selected.
- [ ] Move tools to compact left rail and selection details to right inspector.
- [ ] Preserve pan/zoom/magnifier and review patch generation.
- [ ] Run frontend tests/typecheck/build.

### Task 9: Analysis tabs, results, and export UI

**Files:**
- Modify: `frontend/app/analysis/[id]/page.tsx`
- Create: `frontend/components/ResultsPanel.tsx`
- Create: `frontend/components/ExportPanel.tsx`
- Modify: `frontend/app/globals.css`

**Interfaces:**
- Tabs: `검수`, `결과`, `내보내기`.
- Unsaved review changes are saved before leaving review for results/export.
- Results use active saved review rows; export buttons point to authenticated API download URLs.

- [ ] Build ResultsPanel with summary cards, native-SVG histogram, and 0–180 rose chart using the tested helpers.
- [ ] Build ExportPanel with CSV, measurement image, labeled image, and ZIP download actions.
- [ ] Refactor analysis page into tabs with sticky review save/approve bar and compact top metadata/account controls.
- [ ] Ensure tab changes auto-save pending review changes before displaying results/export.
- [ ] Run frontend tests/typecheck/build.

### Task 10: Configuration, documentation, and release verification

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Keep: existing `.github/workflows/ci.yml` and `.github/workflows/deploy.yml` behavior.

**Interfaces:**
- New settings documented: `AUTH_SESSION_DAYS=7`, `AUTH_COOKIE_NAME=fibervision_session`, `AUTH_COOKIE_SECURE=false`.
- Deployment instructions include one-time user provisioning command.

- [ ] Document auth configuration and `docker compose exec api python -m app.cli.create_user <email>`.
- [ ] Run `PYTHONPATH=backend pytest -q backend/tests`.
- [ ] Run `npm test`, `npm run typecheck`, and `npm run build` in `frontend`.
- [ ] Inspect git diff for secrets, accidental model files, or unrelated refactors.
- [ ] Publish as one main-branch commit only after all available local checks pass; use GitHub CI as the final build verification before automatic deployment.
