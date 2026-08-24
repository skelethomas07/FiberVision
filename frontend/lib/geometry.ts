export type LineGeometry = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type ReviewLine = LineGeometry & {
  id: string;
  source_model_measurement_id: string | null;
  active: boolean;
};

export type ReviewPatch = {
  removed_ids: string[];
  corrected: Array<LineGeometry & { id: string }>;
  added: LineGeometry[];
};

export function lineMetrics(line: LineGeometry) {
  const dx = line.x2 - line.x1;
  const dy = line.y2 - line.y1;
  const width_px = Math.hypot(dx, dy);
  const angle_deg = ((Math.atan2(-dy, dx) * 180) / Math.PI + 180) % 360 - 180;
  return { width_px, angle_deg };
}

function moved(a: ReviewLine, b: ReviewLine, eps = 1e-6) {
  return (
    Math.abs(a.x1 - b.x1) > eps ||
    Math.abs(a.y1 - b.y1) > eps ||
    Math.abs(a.x2 - b.x2) > eps ||
    Math.abs(a.y2 - b.y2) > eps
  );
}

export function buildReviewPatch(initial: ReviewLine[], current: ReviewLine[]): ReviewPatch {
  const initialById = new Map(initial.map((line) => [line.id, line]));
  const removed_ids: string[] = [];
  const corrected: Array<LineGeometry & { id: string }> = [];
  const added: LineGeometry[] = [];

  for (const line of current) {
    const before = initialById.get(line.id);
    if (!before) {
      if (line.active && line.source_model_measurement_id === null) {
        added.push({ x1: line.x1, y1: line.y1, x2: line.x2, y2: line.y2 });
      }
      continue;
    }
    if (before.active && !line.active) {
      removed_ids.push(line.id);
      continue;
    }
    if (line.active && moved(before, line)) {
      corrected.push({ id: line.id, x1: line.x1, y1: line.y1, x2: line.x2, y2: line.y2 });
    }
  }

  removed_ids.sort();
  corrected.sort((a, b) => a.id.localeCompare(b.id));
  return { removed_ids, corrected, added };
}
