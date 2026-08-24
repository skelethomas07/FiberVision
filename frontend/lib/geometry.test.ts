import assert from "node:assert/strict";
import test from "node:test";

import { buildReviewPatch, lineMetrics } from "./geometry.ts";

test("lineMetrics returns pixel width and image-style angle", () => {
  assert.deepEqual(lineMetrics({ x1: 0, y1: 0, x2: 3, y2: 4 }), {
    width_px: 5,
    angle_deg: -53.13010235415598,
  });
});

test("buildReviewPatch separates removed corrected and added lines", () => {
  const initial = [
    { id: "keep", source_model_measurement_id: "m1", x1: 0, y1: 0, x2: 10, y2: 0, active: true },
    { id: "remove", source_model_measurement_id: "m2", x1: 0, y1: 10, x2: 10, y2: 10, active: true },
    { id: "edit", source_model_measurement_id: "m3", x1: 0, y1: 20, x2: 10, y2: 20, active: true },
  ];
  const current = [
    initial[0],
    { ...initial[1], active: false },
    { ...initial[2], x2: 12 },
    { id: "local-1", source_model_measurement_id: null, x1: 0, y1: 30, x2: 8, y2: 30, active: true },
  ];

  assert.deepEqual(buildReviewPatch(initial, current), {
    removed_ids: ["remove"],
    corrected: [{ id: "edit", x1: 0, y1: 20, x2: 12, y2: 20 }],
    added: [{ x1: 0, y1: 30, x2: 8, y2: 30 }],
  });
});

test("buildReviewPatch persists edits to an already-saved manual line", () => {
  const initial = [
    { id: "manual-db", source_model_measurement_id: null, x1: 0, y1: 0, x2: 5, y2: 0, active: true },
  ];
  const current = [{ ...initial[0], x2: 7 }];
  assert.deepEqual(buildReviewPatch(initial, current), {
    removed_ids: [],
    corrected: [{ id: "manual-db", x1: 0, y1: 0, x2: 7, y2: 0 }],
    added: [],
  });
});
