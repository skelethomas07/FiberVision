import assert from "node:assert/strict";
import test from "node:test";

import {
  directionBins,
  fiberAngleFromMeasurement,
  histogramBins,
  historyShortcut,
  normalizeConfidenceThreshold,
  cyclicIndex,
  confidenceMatches,
  measurementStats,
  normalizeFiberAngle,
  sectorBounds,
  sectorGrid,
} from "./reviewView.ts";

test("sectorGrid maps VisionFlux split counts to the expected grids", () => {
  assert.deepEqual(sectorGrid(6), { cols: 3, rows: 2 });
  assert.deepEqual(sectorGrid(9), { cols: 3, rows: 3 });
  assert.deepEqual(sectorGrid(12), { cols: 4, rows: 3 });
  assert.deepEqual(sectorGrid(16), { cols: 4, rows: 4 });
});

test("sectorBounds returns the requested image sector", () => {
  assert.deepEqual(sectorBounds(6, 4, 300, 200), { x: 100, y: 100, width: 100, height: 100 });
  assert.deepEqual(sectorBounds(12, 11, 400, 300), { x: 300, y: 200, width: 100, height: 100 });
});

test("normalizeFiberAngle folds directions into zero through 180 degrees", () => {
  assert.equal(normalizeFiberAngle(0), 0);
  assert.equal(normalizeFiberAngle(190), 10);
  assert.equal(normalizeFiberAngle(-10), 170);
  assert.equal(normalizeFiberAngle(180), 0);
});


test("fiberAngleFromMeasurement rotates a thickness chord into fiber direction", () => {
  assert.equal(fiberAngleFromMeasurement(0), 90);
  assert.equal(fiberAngleFromMeasurement(90), 0);
  assert.equal(fiberAngleFromMeasurement(-45), 45);
});

test("measurementStats returns count mean median and population standard deviation", () => {
  const stats = measurementStats([1, 2, 3, 4]);
  assert.equal(stats.count, 4);
  assert.equal(stats.mean, 2.5);
  assert.equal(stats.median, 2.5);
  assert.ok(Math.abs(stats.stddev - Math.sqrt(1.25)) < 1e-10);
});

test("histogramBins creates stable bins including the maximum value", () => {
  assert.deepEqual(histogramBins([1, 2, 3, 4], 2).map((bin) => bin.count), [2, 2]);
  assert.deepEqual(histogramBins([5, 5, 5], 3).map((bin) => bin.count), [3, 0, 0]);
});

test("directionBins counts normalized fiber orientations", () => {
  assert.deepEqual(directionBins([0, 10, 179, -10, 190], 6).map((bin) => bin.count), [3, 0, 0, 0, 0, 2]);
});


test("historyShortcut uses Ctrl+Z for undo and Ctrl+Y for redo", () => {
  assert.equal(historyShortcut({ key: "z", ctrlKey: true, metaKey: false, shiftKey: false }), "undo");
  assert.equal(historyShortcut({ key: "Y", ctrlKey: true, metaKey: false, shiftKey: false }), "redo");
  assert.equal(historyShortcut({ key: "z", ctrlKey: true, metaKey: false, shiftKey: true }), null);
  assert.equal(historyShortcut({ key: "z", ctrlKey: false, metaKey: false, shiftKey: false }), null);
});

test("cyclicIndex wraps low-confidence navigation", () => {
  assert.equal(cyclicIndex(0, -1, 5), 4);
  assert.equal(cyclicIndex(4, 1, 5), 0);
  assert.equal(cyclicIndex(1, 1, 5), 2);
  assert.equal(cyclicIndex(0, 1, 0), 0);
});

test("confidenceMatches supports all low and normal filters with a configurable threshold", () => {
  assert.equal(confidenceMatches(0.69, "low", 0.7), true);
  assert.equal(confidenceMatches(0.7, "low", 0.7), false);
  assert.equal(confidenceMatches(0.74, "low", 0.75), true);
  assert.equal(confidenceMatches(0.8, "normal", 0.75), true);
  assert.equal(confidenceMatches(0.7, "normal", 0.75), false);
  assert.equal(confidenceMatches(null, "low", 0.75), false);
  assert.equal(confidenceMatches(null, "normal", 0.75), true);
  assert.equal(confidenceMatches(null, "all", 0.75), true);
});

test("normalizeConfidenceThreshold clamps user input to zero through one", () => {
  assert.equal(normalizeConfidenceThreshold(0.8), 0.8);
  assert.equal(normalizeConfidenceThreshold(-1), 0);
  assert.equal(normalizeConfidenceThreshold(2), 1);
  assert.equal(normalizeConfidenceThreshold(Number.NaN), 0.7);
});
