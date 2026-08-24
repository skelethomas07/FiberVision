# Verification

The repository was verified in the sandbox without a trained checkpoint. Heavy inference itself therefore remains a deployment-time checkpoint smoke test; the adapter contract is covered with a fake DataFrame/engine and the extracted model package's post-processing and thick-recovery tests are run directly.

## Commands

```bash
cd backend
PYTHONPATH=. pytest -q tests
PYTHONPATH=. python -m compileall -q app
PYTHONPATH=".:vendor" pytest -q \
  vendor/sem_fiber_ai/tests/test_postprocessing.py \
  vendor/sem_fiber_ai/tests/test_thick_recovery.py
PYTHONPATH=. python -m app.training.cli --help

cd ../frontend
npm test
# Full TSX syntax/local type pass with temporary React/Next declarations because
# this sandbox cannot download npm dependencies.
tsc -p /tmp/tsconfig.frontend-check.json

cd ..
python - <<'PY'
import yaml
with open('docker-compose.yml') as f:
    doc = yaml.safe_load(f)
assert {'postgres','redis','minio','api','worker','frontend'} <= set(doc['services'])
print('compose yaml structure: OK')
PY
```

## Environment limitation

`docker` is not installed in this sandbox, so `docker compose config` and image builds could not be executed here. `next`, `react`, and their type packages are also not preinstalled and npm registry access timed out, so the real `npm run typecheck` / `npm run build` are delegated to `.github/workflows/ci.yml`, where dependencies are installed normally.
