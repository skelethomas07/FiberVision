# sem_fiber_ai 7.0.0

Fibre-width and orientation measurement from SEM micrographs, rebuilt from v6.13
so that every number it prints can be defended.

`VERSION` is the single source of the version string; `src/__init__.py` and every
manifest read it.

## What v7 changes (short form)

* **One angle convention.** Everything internal is raster: `wrap180(atan2(dy, dx))`
  with y pointing down, fibre axis = measurement angle − 90°. ImageJ (y-up) exports
  are converted once, by a fixed transform, and the original value is kept in
  `imagej_angle_deg` with `angle_source_convention` recording what it was. No sign
  or offset is ever fitted from labels.
* **Calibration is never inferred from ground truth.** The annotator-implied scale
  is a contradiction detector only. Fields whose physical scale cannot be resolved
  get `calibration_valid = False` and `width_nm = NaN` — never a guessed number.
* **Sealed, specimen-level splits.** Specimen keys strip exactly one trailing field
  token, near-duplicate images are merged into the same group, and leakage is
  asserted, not assumed.
* **Geometry width representation.** A distance-to-boundary head gives width at the
  ridge as 2×distance, with the sparse-disc baseline kept for comparison.
* **Post-processing is selected on validation only**; the test split is evaluated once.
* **Every site carries a machine-readable `rejected_reason`**, every field a
  PASS / REVIEW / FAIL status, and only PASS fields enter publication summaries.
* **The thick-fibre branch is `experimental` and labelled NOT VALIDATED** until a
  real manual thick-chord table passes the thresholds in `config/default.yaml`.
* **FULL_RUN requires CUDA.** Hardware changes micro-batch, accumulation, workers,
  precision and tiling — never epochs, model, targets, loss, split or gates.

See `V7_CHANGELOG.md`, `VALIDATION_PROTOCOL.md` and `KNOWN_LIMITATIONS.md`
alongside the notebook.

## Layout

```
VERSION               single version string (7.0.0)
config/default.yaml   PROTOCOL sections (hashed into every manifest) + hardware knobs
src/                  the package (see the notebook, section 1, for the map)
tests/                116 tests; run `python -m pytest tests -q`
```

## Quick check

```python
from src.selftest import run_selftest
run_selftest()          # synthetic CPU end-to-end; not a scientific result
```
