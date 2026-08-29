export type SplitCount = 6 | 9 | 12 | 16;

export type SectorRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type NumericBin = {
  start: number;
  end: number;
  count: number;
};

export function sectorGrid(count: SplitCount) {
  switch (count) {
    case 6: return { cols: 3, rows: 2 };
    case 9: return { cols: 3, rows: 3 };
    case 12: return { cols: 4, rows: 3 };
    case 16: return { cols: 4, rows: 4 };
  }
}

export function sectorBounds(count: SplitCount, index: number, imageWidth: number, imageHeight: number): SectorRect {
  const { cols, rows } = sectorGrid(count);
  const safeIndex = Math.max(0, Math.min(count - 1, Math.trunc(index)));
  const col = safeIndex % cols;
  const row = Math.floor(safeIndex / cols);
  return {
    x: (imageWidth * col) / cols,
    y: (imageHeight * row) / rows,
    width: imageWidth / cols,
    height: imageHeight / rows,
  };
}

export function normalizeFiberAngle(angle: number): number {
  return ((angle % 180) + 180) % 180;
}

export function fiberAngleFromMeasurement(angle: number): number {
  return normalizeFiberAngle(angle + 90);
}

export function measurementStats(values: number[]) {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return { count: 0, mean: 0, median: 0, stddev: 0, min: 0, max: 0 };
  const sorted = [...finite].sort((a, b) => a - b);
  const count = sorted.length;
  const mean = sorted.reduce((sum, value) => sum + value, 0) / count;
  const middle = Math.floor(count / 2);
  const median = count % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  const variance = sorted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;
  return {
    count,
    mean,
    median,
    stddev: Math.sqrt(variance),
    min: sorted[0],
    max: sorted[count - 1],
  };
}

export function histogramBins(values: number[], binCount = 12): NumericBin[] {
  const finite = values.filter(Number.isFinite);
  const bins = Math.max(1, Math.trunc(binCount));
  if (!finite.length) return [];
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (min === max) {
    const width = Math.max(Math.abs(min) * 0.05, 1);
    return Array.from({ length: bins }, (_, index) => ({
      start: min + index * width,
      end: min + (index + 1) * width,
      count: index === 0 ? finite.length : 0,
    }));
  }
  const width = (max - min) / bins;
  const result = Array.from({ length: bins }, (_, index) => ({
    start: min + index * width,
    end: min + (index + 1) * width,
    count: 0,
  }));
  for (const value of finite) {
    const index = value === max ? bins - 1 : Math.min(bins - 1, Math.floor((value - min) / width));
    result[index].count += 1;
  }
  return result;
}

export function directionBins(angles: number[], binCount = 12): NumericBin[] {
  const bins = Math.max(1, Math.trunc(binCount));
  const width = 180 / bins;
  const result = Array.from({ length: bins }, (_, index) => ({ start: index * width, end: (index + 1) * width, count: 0 }));
  for (const angle of angles.filter(Number.isFinite)) {
    const normalized = normalizeFiberAngle(angle);
    const index = Math.min(bins - 1, Math.floor(normalized / width));
    result[index].count += 1;
  }
  return result;
}
