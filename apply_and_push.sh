#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/home/ubuntu/FiberVision}"
EXPECTED_SHA="f4e7becc0a4193e14b18a9e96d12428c935994b2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$SCRIPT_DIR/payload"

if [[ ! -d "$REPO/.git" ]]; then
  echo "ERROR: Git repository not found: $REPO" >&2
  exit 1
fi

cd "$REPO"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Repository has uncommitted changes. Nothing was changed." >&2
  git status --short
  exit 1
fi

CURRENT_SHA="$(git rev-parse HEAD)"
if [[ "$CURRENT_SHA" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: Expected main SHA $EXPECTED_SHA but found $CURRENT_SHA." >&2
  echo "The repository changed after this update bundle was prepared. Stop and ask ChatGPT to rebase the bundle." >&2
  exit 1
fi

FILES=(
  ".env.example"
  ".gitignore"
  "README.md"
  "backend/app/api/auth.py"
  "backend/app/api/exports.py"
  "backend/app/api/reviews.py"
  "backend/app/cli/__init__.py"
  "backend/app/cli/create_user.py"
  "backend/app/config.py"
  "backend/app/main.py"
  "backend/app/models.py"
  "backend/app/services/auth.py"
  "backend/app/services/exports.py"
  "backend/requirements-api.txt"
  "backend/tests/test_api.py"
  "backend/tests/test_auth_api.py"
  "backend/tests/test_auth_service.py"
  "backend/tests/test_create_user_cli.py"
  "backend/tests/test_exports.py"
  "docs/superpowers/plans/2026-08-30-auth-review-results-export.md"
  "docs/superpowers/specs/2026-08-30-auth-review-results-export-design.md"
  "frontend/app/analysis/[id]/page.tsx"
  "frontend/app/change-password/page.tsx"
  "frontend/app/globals.css"
  "frontend/app/layout.tsx"
  "frontend/app/login/page.tsx"
  "frontend/app/page.tsx"
  "frontend/components/AuthGuard.tsx"
  "frontend/components/ExportPanel.tsx"
  "frontend/components/MeasurementCanvas.tsx"
  "frontend/components/ResultsPanel.tsx"
  "frontend/components/UserMenu.tsx"
  "frontend/lib/api.test.ts"
  "frontend/lib/api.ts"
  "frontend/lib/reviewView.test.ts"
  "frontend/lib/reviewView.ts"
)

for file in "${FILES[@]}"; do
  mkdir -p "$(dirname "$REPO/$file")"
  cp "$PAYLOAD/$file" "$REPO/$file"
done

git diff --check
git add -- "${FILES[@]}"

echo
echo "Files staged:"
git status --short

git commit -m "Add authentication review results and exports"
git push origin main

echo
echo "Pushed to GitHub main."
echo "GitHub Actions will run CI first; EC2 deployment runs only if CI succeeds."
