# v6.6 wide-fibre recovery

v6.6 keeps the v6.5 trained network/checkpoint unchanged and adds an opt-in
inference supplement (`src/thick_fiber.py`) for broad fibres that are visible in
the dense segmentation map but under-sampled by the learned centre heatmap.

The supplement combines a gamma-normalised multi-scale Hessian bank, an Otsu
fibre-body EDT/medial axis, and an across-fibre intensity-profile width.  It
merges these rows with the learned output and only replaces narrow,
same-orientation detections on the recovered wide fibre.  CSV rows expose
`measurement_source` and the scale/EDT/profile diagnostics.

CLI: add `--thick_recovery`.  Existing evaluation remains pure-AI unless that
flag is explicitly supplied.

# sem_fiber_ai

Automatic detection and thickness measurement of fibers in SEM images,
reproducing manual ImageJ-style measurements.

The model takes an unannotated SEM image and returns, for every measurable site
it finds: a measurement line across the fiber, its centre and endpoints, the
local fiber orientation, the measurement-line orientation, thickness in pixels
and (when the scale is known) in nanometres, and a confidence and uncertainty
estimate — plus an annotated image, a CSV, summary statistics and a thickness
histogram.

---

## Status

The data-handling half of this project has been built **and run on real data**
(one annotated field, `2-21`, with 550 manual measurements). The model half is
complete and unit-tested for shapes, gradients and masking, but has **not been
trained**, because the dataset as supplied cannot support training. See
[Limitations](#limitations).

Do not treat any number this repository produces as a validated accuracy until
it has been measured on SEM images the model never saw.

---

## What the audit found in the supplied data

These are findings from the actual files, not assumptions.

**Pixel size.** For `2-11` two independent sources agree: the burned-in footer
reads `FOV:1280x960nm` over a 1280 px-wide frame → 1.000 nm/px, and the scale
bar measures 99 px for 100 nm → 1.01 nm/px. The footer is located
geometrically (a run of near-black rows at the bottom), so this works for any
frame size, not just 1280×1024.

**Units of the `Length` column.** 98.4 % of the 550 values sit on an exact
integer lattice of step 16/15 = 1.0666667. That is the fingerprint of a
measurement taken in whole pixels on an image calibrated at 1.0667 units/px —
i.e. a 1200×900 px source covering a 1280×960 nm field. The supplied PNG is
1280×960, so on *that* file 1 px = 1 nm and `Length` in nm is numerically equal
to a width in PNG pixels. `infer_length_quantum()` reports this lattice and the
fraction of values it explains, so the same check runs on any new export.

**Marker and angle conventions.** Neither is knowable from the files, and both
are catastrophic to get wrong, so `calibrate_marker_geometry()` decides them
from the image: it scans candidate marker offsets and both angle sign
conventions, and keeps whichever makes the pooled intensity profile across the
chords symmetric and bright in the middle. On `2-21` the answer is an offset of
(−5, +5) px with the raster (y-down) sign convention, giving a profile that is
flat at ~113 grey levels outside |t| > 0.6, rises to 151 at the centre, and is
symmetric to within 1.6 grey levels. The median chord FWHM is 1.05 × the
reported width — the reported `Length` is the full fiber thickness.

**Positions.** The CSV has no coordinates, so positions were recovered from the
overlay. 404 of 550 (73 %) were recovered; every miss is logged with a reason.

---

## Two extraction paths, and which one you want

**Preferred: the line overlay.** When the export draws the measurement *chords*
(`<id>_thickness.png`) rather than only numbered markers, the geometry is
already in the image. Each connected chord yields its centre, orientation and
length directly; a Hungarian assignment on (length, angle) pairs each chord to
its CSV row, so **no OCR is involved at all**. On `3-8`: 739 chords found,
729 of 767 rows matched (95%), median length residual **0.25 px**, angle
residual 1.8 deg, r = 0.987.

If both renderings are supplied, the pipeline tests each and keeps whichever
yields more chords -- the labelled version paints boxes over the lines and
loses most of them, so filenames are not trusted.

Three conventions are then settled from the data rather than assumed, each by
the same principle: try the alternatives, keep the one the image supports.

| unknown | how it is decided | result on `3-8` |
|---|---|---|
| units of `Length` | which interpretation reproduces the drawn pixel lengths | **nm** (729 matched vs 20 for pixels) |
| angle sign convention | which sign matches the drawn orientations | **y-up** (1.8 deg vs 20.0 deg) |
| structure-tensor orientation mapping | which makes chords perpendicular to ridges | sign −1, offset 90 deg |

That last one doubles as an end-to-end validation: after alignment the manual
chords sit **4.2 deg** from perpendicular to the local ridge, with 100% within
30 deg. If any earlier step were wrong, that number would be near 45 deg.

Note the units differ between export types in the same project: the VisionFlux
`*_ImageJ_results.csv` files are in **nanometres**, the manual ImageJ
`*_Results.csv` files are in **pixels**. This is exactly why nothing is
hard-coded.

## Fallback: how the numbers were read off a marker-only overlay

The measurement ids are printed in an 8-pixel-tall bitmap font, which defeats
general-purpose OCR (Tesseract managed 361/550 reads, only 230 of them unique —
worse than useless). But a deterministic bitmap font has a property worth
exploiting: **every instance of a digit is pixel-identical**. So:

1. Segment all glyphs and cluster them by exact bitmap. This yields exactly ten
   clusters — one per digit.
2. Name the clusters once, using Tesseract on unambiguous single-digit boxes and
   then on the isolated template bitmaps rendered large. Any cluster still
   unnamed is resolved by testing which assignment maximises the count of
   unique, in-range ids.
3. Locate every digit by **exact tiling**: within each text row, find a
   left-to-right cover of the row's ink by exact template matches, with no ink
   left over and no overlap.

Step 3 is deliberately stricter than template correlation, for two reasons that
both corrupt labels silently:

* the 3-px-wide `1` correlates above 0.9 against the vertical stroke inside `4`,
  `7` and even a box edge, inventing ids that never existed;
* digits are sometimes rendered touching, so anything that demands a blank
  separator truncates `681` to `68` — which then steals label 68 from its real
  owner.

This is high precision and moderate recall by design: rows it cannot tile
exactly are reported unreadable rather than guessed at.

The remaining ids are then filled by `complete_labels_by_order()`. On this
dataset label id turns out to increase monotonically with marker *y* — but that
is **tested on the exactly-read labels first** (inversion rate 0.166, corr 0.999)
and only applied if it holds. Where it holds, a run of unlabelled markers between
two anchors must carry exactly the ids between them; when the counts match the
assignment is forced and unique, so no guessing is involved. When they don't
match, the run is left unassigned and reported.

---

## Why local measurement rather than instance segmentation

In a phase-separated or electrospun membrane the fibers form a deeply
overlapping 3-D network projected onto 2-D. A single fiber is occluded and
re-emerges many times, and at a crossing there is genuinely no image evidence for
which strand passes in front. Asking a model for *instances* poses a question the
pixels cannot answer: instance identity is unobservable, so the labels would
encode annotator convention and the metric would measure that convention rather
than any physical quantity.

Thickness is a *local* property. At a point on a fiber, diameter is well defined
from the local intensity ridge alone. Predicting dense per-pixel quantities — is
this a fiber, which way does it run, how thick is it, is this a place where a
reliable measurement can be made — asks only answerable questions, and it matches
how the manual measurements were produced: a human picks a clean spot and draws
one chord. The centre heatmap therefore learns the human's *site-selection
policy*, which is exactly the behaviour to reproduce.

### Model outputs

| head | channels | meaning |
|---|---|---|
| `center_logit` | 1 | Gaussian peak at each measurable site |
| `segment_logit` | 1 | fiber presence |
| `orient` | 2 | (cos 2θ, sin 2θ) — π-periodic fiber direction |
| `width` | 1 | log fiber thickness in pixels |
| `validity_logit` | 1 | is a reliable measurement possible here |
| `logvar` | 1 | aleatoric uncertainty on the width |

Orientation is encoded on the doubled angle so the loss is π-periodic by
construction: a fiber at +85° and one at −85° are 10° apart, not 170°.

At inference: peak detection → NMS → read width/orientation/σ at each peak →
build the chord perpendicular to the local fiber axis → reject implausible or
low-validity detections → suppress duplicates. That last step suppresses only
when the predicted *orientations* also agree, because two peaks 20 px apart on
the same fiber are duplicates while two peaks 20 px apart on two crossing fibers
are both legitimate.

---

## Install

```bash
pip install -r requirements.txt          # or: conda env create -f environment.yml
# optional, enables reading the burned-in SEM footer for calibration:
#   apt-get install tesseract-ocr    /    brew install tesseract
```

Runs on Colab, Linux and Windows; CUDA is used when available with automatic
mixed precision, and inference falls back to CPU. Everything except training and
inference (audit, extraction, verification) runs without PyTorch.

## Organise your data

```
data/
├── original/     clean SEM images, footer intact   2-21.jpg
├── annotated/    overlays with markers + numbers   2-21_labeled_thickness.png
├── csv/          ImageJ results tables             2-21_ImageJ_results.csv
└── processed/    written by the pipeline
```

Files are paired by image id, derived from the filename stem with common
suffixes stripped (`2-21_labeled_thickness.png` → `2-21`). **Keep the footer on
the originals** — it is where the pixel size comes from.

## Run

```bash
# 1. audit first, always
python -m sem_fiber_ai.src.audit_data --data_dir data/

# 2. recover annotations and verify them by eye
python -m sem_fiber_ai.src.annotation_extraction \
    --original_dir data/original --annotated_dir data/annotated \
    --csv_dir data/csv --output_csv data/processed/labels.csv \
    --debug_dir outputs/debug

# 3. baseline sanity check, then the real model
python -m sem_fiber_ai.src.train --config config/default.yaml --model baseline
python -m sem_fiber_ai.src.train --config config/default.yaml --model full

# 4. evaluate on whole unseen images
python -m sem_fiber_ai.src.evaluate --checkpoint outputs/best_full.pt --split test

# 5. predict on new images
python -m sem_fiber_ai.src.infer --checkpoint outputs/best_full.pt \
    --image new_sem_image.jpg --nm_per_pixel 2.0 --output_dir predictions/
python -m sem_fiber_ai.src.infer --checkpoint outputs/best_full.pt \
    --image_dir data/new --output_dir predictions/
```

Notebooks in `notebooks/` mirror these four stages for Colab.

## Tests

```bash
python -m pytest tests -q
```

50 tests covering CSV schema inference, coordinate/angle round-trips,
augmentation invariants (width and nm/px scale together; endpoints stay
consistent with the angle), calibration refusal behaviour, decoding, duplicate
suppression and Hungarian matching. `test_model_smoke.py` adds shape, gradient,
masking and tiled-inference checks; it skips automatically when torch is absent.

---

## Design decisions worth knowing about

**Leakage.** Splitting is always by `image_id`, never by annotation or patch.
Fiber networks are locally self-similar, so two patches from one field are far
more alike than two patches from different specimens; a random patch split
would report excellent metrics that mean nothing. Perceptually near-identical
images are merged into one group before splitting, and `evaluate.py` refuses to
run if the requested split overlaps train.

**Augmentation.** Anisotropic resizing is forbidden — it changes apparent fiber
thickness as a function of orientation. Isotropic scaling is allowed, and the
width labels *and* `nm_per_pixel` are scaled with it, so physical size is
invariant (there is a test for this). Angles are recovered from the transformed
endpoints rather than tracked separately, so the transform matrix is the single
source of truth.

**Calibration.** Resolution order is: explicit argument → sidecar `.calib.json`
or table → footer FOV text → scale bar → **nothing**. There is no default. When
the scale is unknown, thickness is reported in pixels and the nm column is NaN.

**Model size.** The U-Net is deliberately small (`base: 32`, `depth: 4`). With a
handful of labelled images a 30 M-parameter pretrained backbone would memorise
the training field long before it learned anything transferable. A timm encoder
sits behind a config flag for when the dataset grows.

**Uncertainty.** The width head is trained with a heteroscedastic Gaussian NLL,
so the model can say "this is a crossing, my estimate is unreliable" rather than
being forced to commit. Test-time augmentation and MC-dropout are available but
off by default — per the brief, complexity is added only when it earns its keep
on validation.

**Advanced options not enabled.** Self-supervised pretraining, pseudo-labelling,
teacher–student consistency, active learning and deep ensembles are all
reasonable next steps and none of them is turned on. With one labelled image
there is no validation signal to demonstrate they help, and unmeasured
complexity is a liability.

---

## Limitations

Read this before drawing conclusions.

1. **No complete training triplet exists in the supplied data.** The CSV and
   overlay are both `2-21`; the only clean image is `2-11`, a different field.
   The pipeline can reconstruct a clean `2-21` by inpainting, but 38 % of that
   image is overlay, and inpainted pixels are not evidence. Supply the clean
   `2-21` with its footer.

2. **One field means no generalisation claim is possible.** A grouped split
   needs at least three independent images. `grouped_split()` says so loudly and
   `train.py` marks the run `proof_of_concept: true` in its manifest. For a
   defensible result: roughly **15–30 labelled fields** spanning your
   magnifications and sample groups, with **5–8 held out entirely**.

3. **Recovered positions are good in aggregate, not per measurement.** The
   pooled profile is textbook (contrast 22.9, asymmetry 1.6, median FWHM 1.05 ×
   reported width), but only ~44 % of *individual* annotations show a FWHM
   within ±30 % of their reported width. That is good enough to demonstrate the
   pipeline and not good enough to regress against.

   **This is fixable upstream in about a minute.** In ImageJ:
   `Analyze ▸ Set Measurements ▸ Bounding rectangle + Centroid`, or
   `List coordinates` from the ROI Manager. If the overlay came from your own
   export script, write the centre and endpoints into the CSV. That removes the
   OCR, the marker-offset inference, the angle-sign ambiguity and the 27 %
   recovery loss all at once — `csv_parser.py` already detects and uses
   coordinate columns whenever they are present.

4. **The torch modules are unit-tested but were never trained.** No GPU run has
   happened. Run `pytest tests/test_model_smoke.py` first in your environment:
   it catches head-shape and loss-masking errors in seconds.

## Repository layout

```
sem_fiber_ai/
├── config/default.yaml          every knob, commented with its rationale
├── notebooks/                   01 audit · 02 verify · 03 train · 04 infer
├── src/
│   ├── audit_data.py            dataset report + headline warnings
│   ├── csv_parser.py            schema inference, unit lattice detection
│   ├── calibration.py           nm/px with provenance, never a default
│   ├── image_registration.py    identity → ECC → ORB, similarity only
│   ├── annotation_extraction.py markers, glyph OCR, geometry calibration
│   ├── targets.py               sparse measurements → dense maps + masks
│   ├── augmentations.py         geometry-aware, thickness-preserving
│   ├── dataset.py               tiles, patches, leakage-safe splitting
│   ├── models/                  patch baseline + multi-head U-Net
│   ├── losses.py                focal · dice+BCE · heteroscedastic · vector
│   ├── postprocess.py           peaks → chords, orientation-aware NMS
│   ├── matching.py              Hungarian one-to-one assignment
│   ├── metrics.py               detection · thickness · angle · calibration
│   ├── visualization.py         verification, panels, error maps, BA plots
│   ├── train.py / evaluate.py / infer.py
│   └── utils.py
├── tests/                       50 tests + torch smoke tests
└── outputs/
```
