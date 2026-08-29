import assert from "node:assert/strict";
import test from "node:test";

import {
  directionBins,
  fiberAngleFromMeasurement,
  histogramBins,
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
