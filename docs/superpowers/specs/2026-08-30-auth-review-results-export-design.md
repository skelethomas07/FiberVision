# FiberVision Auth, Review Workspace, Results, and Export Design

## Goal

Turn the existing FiberVision MVP into a clean, access-controlled SEM review workspace while preserving the v6.11 inference, VisionFlux import, TIFF preview, human review, and training-example flow.

## Product flow

1. User visits FiberVision and signs in with an administrator-created email/password account.
2. First login requires replacing the initial password before any analysis API can be used.
3. Home screen is a minimal upload workspace for raw SEM or VisionFlux-annotated JPG/PNG/TIFF images.
4. Analysis opens a three-tab workspace: `검수`, `결과`, `내보내기`.
5. Review provides VisionFlux-style fast editing with a professional analysis-tool layout.
6. Review approval continues to create training examples exactly as today.

## Authentication

- No self-signup and no web admin console.
- Administrators create accounts on EC2 with `python -m app.cli.create_user <email>` inside the API container.
- Passwords are stored only as Argon2id hashes.
- Login creates an opaque random session token. Only the SHA-256 hash of that token is stored in PostgreSQL.
- Browser receives the token in an HttpOnly, SameSite=Lax cookie. Secure is configurable and remains off while the service is HTTP-only.
- Sessions expire after seven days by default.
- Disabled, missing, expired, or unknown sessions receive 401.
- Users with `must_change_password=true` may access only auth endpoints until they change the initial password.
- All image, analysis, review, result, and export APIs require authentication. `/healthz` and `/api/auth/*` remain public as needed.

## Review workspace

Desktop layout:

- top application bar: FiberVision, analysis metadata, current account, new analysis/logout actions
- tabs: `검수`, `결과`, `내보내기`
- review tab: left compact tool rail, center SEM viewer, right selected-measurement inspector
- bottom sticky review bar: change count, save, approve

Tools use short Korean names: `선택`, `추가`, `수정`, `삭제`, `돋보기`, `분할`, `맞춤`.

Existing zoom, pan, endpoint editing, line adding/deleting, magnifier, TIFF preview, and VisionFlux color semantics remain.

### Split review

Supported split counts match VisionFlux:

- 6 = 3 columns × 2 rows
- 9 = 3 × 3
- 12 = 4 × 3
- 16 = 4 × 4

A selected split shows `current / total`, previous/next controls, and `맞춤` fits the current sector exactly into the viewer. `전체` returns to whole-image fit.

### Review helpers

- Undo/redo for local line add/delete/correction actions.
- Measurement overlay show/hide.
- Filter: all or low confidence; low confidence means model confidence below 0.70. Manual lines are not treated as low-confidence model predictions.
- Inspector shows width, angle, source, confidence, and edit state for the selected line.

## Results

Results remain separate from review to avoid clutter. Calculations use the currently saved review measurements and active lines only.

- summary: count, mean, median, standard deviation
- thickness histogram using nm when calibration exists, otherwise px
- orientation rose chart using fiber orientation normalized to 0–180 degrees
- compact source filter: all / automatic / manual / corrected

Charts use lightweight native SVG rather than adding a chart library.

## Export

Exports are generated from saved active review measurements, not raw model predictions.

- CSV: index, coordinates, width_px, width_nm, angle_deg, source, edited/status
- overlay PNG: final active measurement lines on the SEM preview
- labeled PNG: overlay plus stable numeric labels matching CSV row numbers
- ZIP: CSV + overlay PNG + labeled PNG

Automatic/VisionFlux lines use yellow and manual additions use blue. Selected-state red is UI-only and never exported.

## Home and visual design

- neutral light shell, restrained borders and shadows, compact typography
- dark SEM viewer is the main visual focus
- no decorative stock photography; the application should feel like a professional microscopy/process-analysis tool
- upload screen uses one prominent drop zone and minimal explanatory copy
- responsive behavior collapses inspector/tool rail appropriately on narrow screens

## Compatibility and deployment

- Existing analysis/review/training tables remain unchanged; authentication adds new tables only, avoiding migrations of existing production tables.
- FastAPI continues to serve API on 8000 and Next.js on 3000.
- Frontend API calls include credentials so the API session cookie works across the two ports on the same host.
- GitHub CI must pass backend pytest, frontend unit tests, typecheck, and build before the existing workflow deploys main through AWS SSM.
