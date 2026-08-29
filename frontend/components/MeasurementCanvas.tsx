"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type WheelEvent as ReactWheelEvent } from "react";
import { buildReviewPatch, lineMetrics, type ReviewPatch } from "@/lib/geometry";
import { cyclicIndex, fiberAngleFromMeasurement, historyShortcut, sectorBounds, type SplitCount } from "@/lib/reviewView";
import type { ReviewMeasurement } from "@/lib/api";

type CanvasLine = ReviewMeasurement;
type ToolMode = "select" | "add" | "edit" | "delete";
type Point = { x: number; y: number };
type DragState =
  | { kind: "pan"; start: Point; origin: Point }
  | { kind: "endpoint"; id: string; endpoint: 1 | 2; before: CanvasLine[] }
  | { kind: "add"; start: Point; current: Point }
  | null;

type Props = {
  imageUrl: string;
  measurements: CanvasLine[];
  disabled?: boolean;
  onPatchChange: (patch: ReviewPatch) => void;
};

const LENS_RADIUS = 88;
const LENS_ZOOM = 2.5;
const LOW_CONFIDENCE = 0.7;
const SPLITS: SplitCount[] = [6, 9, 12, 16];

function cloneLines(lines: CanvasLine[]) {
  return lines.map((line) => ({ ...line }));
}

function segmentDistance(p: Point, line: CanvasLine) {
  const vx = line.x2 - line.x1;
  const vy = line.y2 - line.y1;
  const wx = p.x - line.x1;
  const wy = p.y - line.y1;
  const len2 = vx * vx + vy * vy;
  const t = len2 ? Math.max(0, Math.min(1, (wx * vx + wy * vy) / len2)) : 0;
  return Math.hypot(p.x - (line.x1 + t * vx), p.y - (line.y1 + t * vy));
}

function lineColor(line: CanvasLine, selected: boolean) {
  if (selected) return "#ff556f";
  if (line.source === "manual" || line.source === "visionflux_manual" || line.source_model_measurement_id === null) return "#51a9ff";
  return "#ffd34d";
}

function sourceLabel(line: CanvasLine) {
  if (line.source_model_measurement_id === null || line.source === "manual" || line.source === "visionflux_manual") return "수동 추가";
  if (line.edited) return "수정됨";
  return "자동 측정";
}

function ToolButton({ active, disabled, label, glyph, onClick }: { active?: boolean; disabled?: boolean; label: string; glyph: string; onClick: () => void }) {
  return (
    <button className={`tool-button ${active ? "active" : ""}`} disabled={disabled} onClick={onClick} title={label}>
      <span className="tool-glyph" aria-hidden="true">{glyph}</span>
      <span>{label}</span>
    </button>
  );
}

export default function MeasurementCanvas({ imageUrl, measurements, disabled = false, onPatchChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const initialRef = useRef<CanvasLine[]>(cloneLines(measurements));

  const [lines, setLines] = useState<CanvasLine[]>(cloneLines(measurements));
  const [past, setPast] = useState<CanvasLine[][]>([]);
  const [future, setFuture] = useState<CanvasLine[][]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mode, setMode] = useState<ToolMode>("select");
  const [magnifierOn, setMagnifierOn] = useState(true);
  const [overlayOn, setOverlayOn] = useState(true);
  const [lowConfidenceOnly, setLowConfidenceOnly] = useState(false);
  const [hoverPoint, setHoverPoint] = useState<Point | null>(null);
  const [drag, setDrag] = useState<DragState>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 });
  const [size, setSize] = useState({ width: 900, height: 640 });
  const [splitCount, setSplitCount] = useState<SplitCount | null>(null);
  const [sectorIndex, setSectorIndex] = useState(0);

  useEffect(() => {
    const next = cloneLines(measurements);
    initialRef.current = next;
    setLines(next);
    setPast([]);
    setFuture([]);
    setSelectedId(null);
  }, [measurements]);

  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: Math.max(480, entry.contentRect.height) });
    });
    observer.observe(shell);
    return () => observer.disconnect();
  }, []);

  const fitRect = useCallback((rect: { x: number; y: number; width: number; height: number }, padding = 0.97) => {
    if (!rect.width || !rect.height) return;
    const nextScale = Math.min(size.width / rect.width, size.height / rect.height) * padding;
    setScale(nextScale);
    setOffset({
      x: size.width / 2 - (rect.x + rect.width / 2) * nextScale,
      y: size.height / 2 - (rect.y + rect.height / 2) * nextScale,
    });
  }, [size]);

  const fitWhole = useCallback(() => {
    const image = imageRef.current;
    if (!image?.naturalWidth || !image.naturalHeight) return;
    fitRect({ x: 0, y: 0, width: image.naturalWidth, height: image.naturalHeight }, 0.96);
  }, [fitRect]);

  const fitCurrentSector = useCallback((count = splitCount, index = sectorIndex) => {
    const image = imageRef.current;
    if (!image?.naturalWidth || !image.naturalHeight) return;
    if (!count) {
      fitWhole();
      return;
    }
    fitRect(sectorBounds(count, index, image.naturalWidth, image.naturalHeight), 0.995);
  }, [fitRect, fitWhole, sectorIndex, splitCount]);

  useEffect(() => {
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      window.requestAnimationFrame(() => fitWhole());
    };
    image.onerror = () => { imageRef.current = null; };
    image.src = imageUrl;
    return () => { imageRef.current = null; };
  }, [imageUrl, fitWhole]);

  useEffect(() => { onPatchChange(buildReviewPatch(initialRef.current, lines)); }, [lines, onPatchChange]);

  const calibration = useMemo(() => {
    const line = measurements.find((item) => item.width_nm != null && item.width_px > 0);
    return line?.width_nm != null ? line.width_nm / line.width_px : null;
  }, [measurements]);

  const screenToWorld = useCallback((point: Point) => ({ x: (point.x - offset.x) / scale, y: (point.y - offset.y) / scale }), [offset, scale]);
  const worldToScreen = useCallback((point: Point) => ({ x: offset.x + point.x * scale, y: offset.y + point.y * scale }), [offset, scale]);

  const visibleLines = useMemo(() => lines.filter((line) => {
    if (!line.active || !overlayOn) return false;
    if (!lowConfidenceOnly) return true;
    return line.confidence != null && line.confidence < LOW_CONFIDENCE;
  }), [lines, lowConfidenceOnly, overlayOn]);

  const lowConfidenceLines = useMemo(
    () => lines.filter((line) => line.active && line.confidence != null && line.confidence < LOW_CONFIDENCE),
    [lines],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${size.height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.fillStyle = "#090d14";
    ctx.fillRect(0, 0, size.width, size.height);
    ctx.drawImage(image, offset.x, offset.y, image.naturalWidth * scale, image.naturalHeight * scale);

    if (splitCount) {
      const rect = sectorBounds(splitCount, sectorIndex, image.naturalWidth, image.naturalHeight);
      const topLeft = worldToScreen({ x: rect.x, y: rect.y });
      ctx.save();
      ctx.strokeStyle = "rgba(93, 210, 224, .8)";
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 5]);
      ctx.strokeRect(topLeft.x, topLeft.y, rect.width * scale, rect.height * scale);
      ctx.restore();
    }

    const drawLines = (mapper: (point: Point) => Point, selectedHandles = true) => {
      for (const line of visibleLines) {
        const a = mapper({ x: line.x1, y: line.y1 });
        const b = mapper({ x: line.x2, y: line.y2 });
        const selected = line.id === selectedId;
        ctx.strokeStyle = lineColor(line, selected);
        ctx.lineWidth = selected ? 3 : 1.7;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        if (selected && selectedHandles && mode === "edit") {
          ctx.fillStyle = "#ffffff";
          for (const point of [a, b]) {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    };
    drawLines(worldToScreen);

    if (drag?.kind === "add") {
      const a = worldToScreen(drag.start);
      const b = worldToScreen(drag.current);
      ctx.strokeStyle = "#51a9ff";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (magnifierOn && hoverPoint && !drag) {
      const world = screenToWorld(hoverPoint);
      const lensMap = (point: Point) => ({
        x: hoverPoint.x + (point.x - world.x) * scale * LENS_ZOOM,
        y: hoverPoint.y + (point.y - world.y) * scale * LENS_ZOOM,
      });
      ctx.save();
      ctx.beginPath();
      ctx.arc(hoverPoint.x, hoverPoint.y, LENS_RADIUS, 0, Math.PI * 2);
      ctx.clip();
      ctx.fillStyle = "#090d14";
      ctx.fillRect(hoverPoint.x - LENS_RADIUS, hoverPoint.y - LENS_RADIUS, LENS_RADIUS * 2, LENS_RADIUS * 2);
      ctx.drawImage(
        image,
        hoverPoint.x - world.x * scale * LENS_ZOOM,
        hoverPoint.y - world.y * scale * LENS_ZOOM,
        image.naturalWidth * scale * LENS_ZOOM,
        image.naturalHeight * scale * LENS_ZOOM,
      );
      drawLines(lensMap, false);
      ctx.restore();
      ctx.beginPath();
      ctx.arc(hoverPoint.x, hoverPoint.y, LENS_RADIUS, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,.9)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [drag, hoverPoint, magnifierOn, mode, offset, scale, screenToWorld, sectorIndex, size, splitCount, visibleLines, worldToScreen, selectedId]);

  const commitLines = useCallback((next: CanvasLine[]) => {
    if (disabled) return;
    setPast((history) => [...history.slice(-49), cloneLines(lines)]);
    setLines(next);
    setFuture([]);
  }, [disabled, lines]);

  const undo = useCallback(() => {
    if (!past.length || disabled) return;
    const previous = past[past.length - 1];
    setPast((history) => history.slice(0, -1));
    setFuture((history) => [cloneLines(lines), ...history].slice(0, 50));
    setLines(cloneLines(previous));
    setSelectedId(null);
  }, [disabled, lines, past]);

  const redo = useCallback(() => {
    if (!future.length || disabled) return;
    const next = future[0];
    setFuture((history) => history.slice(1));
    setPast((history) => [...history.slice(-49), cloneLines(lines)]);
    setLines(cloneLines(next));
    setSelectedId(null);
  }, [disabled, future, lines]);

  const pointer = (event: ReactPointerEvent<HTMLCanvasElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const nearestLine = useCallback((screen: Point) => {
    const world = screenToWorld(screen);
    return visibleLines
      .map((line) => ({ line, d: segmentDistance(world, line) * scale }))
      .sort((a, b) => a.d - b.d)[0];
  }, [scale, screenToWorld, visibleLines]);

  const removeLine = useCallback((id: string) => {
    const next = lines.flatMap((line) => line.id !== id ? [line] : line.id.startsWith("local-") ? [] : [{ ...line, active: false }]);
    commitLines(next);
    setSelectedId((current) => current === id ? null : current);
  }, [commitLines, lines]);

  const focusLine = useCallback((line: CanvasLine) => {
    const nextScale = Math.max(scale, 0.8);
    const middle = { x: (line.x1 + line.x2) / 2, y: (line.y1 + line.y2) / 2 };
    setScale(nextScale);
    setOffset({ x: size.width / 2 - middle.x * nextScale, y: size.height / 2 - middle.y * nextScale });
    setSelectedId(line.id);
    setMode("select");
  }, [scale, size]);

  const moveLowConfidence = useCallback((delta: number, startAtFirst = false) => {
    if (!lowConfidenceLines.length) return;
    const current = lowConfidenceLines.findIndex((line) => line.id === selectedId);
    const base = startAtFirst ? -1 : current >= 0 ? current : delta > 0 ? -1 : 0;
    const index = cyclicIndex(base, delta, lowConfidenceLines.length);
    focusLine(lowConfidenceLines[index]);
  }, [focusLine, lowConfidenceLines, selectedId]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      const history = historyShortcut(event);
      if (history === "undo") {
        event.preventDefault();
        undo();
        return;
      }
      if (history === "redo") {
        event.preventDefault();
        redo();
        return;
      }
      if (event.key === "Delete" && selectedId && !disabled) {
        event.preventDefault();
        removeLine(selectedId);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [disabled, redo, removeLine, selectedId, undo]);

  function onPointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    const screen = pointer(event);
    const world = screenToWorld(screen);

    if (!disabled && mode === "delete") {
      const hit = nearestLine(screen);
      if (hit && hit.d < 11) removeLine(hit.line.id);
      return;
    }
    if (!disabled && mode === "add") {
      setDrag({ kind: "add", start: world, current: world });
      return;
    }

    if (!disabled && mode === "edit" && selectedId) {
      const selected = lines.find((line) => line.id === selectedId && line.active);
      if (selected) {
        const a = worldToScreen({ x: selected.x1, y: selected.y1 });
        const b = worldToScreen({ x: selected.x2, y: selected.y2 });
        if (Math.hypot(screen.x - a.x, screen.y - a.y) < 13) {
          setDrag({ kind: "endpoint", id: selected.id, endpoint: 1, before: cloneLines(lines) });
          return;
        }
        if (Math.hypot(screen.x - b.x, screen.y - b.y) < 13) {
          setDrag({ kind: "endpoint", id: selected.id, endpoint: 2, before: cloneLines(lines) });
          return;
        }
      }
    }

    const hit = nearestLine(screen);
    if (hit && hit.d < 9) {
      setSelectedId(hit.line.id);
      return;
    }
    setSelectedId(null);
    setDrag({ kind: "pan", start: screen, origin: offset });
  }

  function onPointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    const screen = pointer(event);
    setHoverPoint(screen);
    if (!drag) return;
    const world = screenToWorld(screen);
    if (drag.kind === "pan") {
      setOffset({ x: drag.origin.x + screen.x - drag.start.x, y: drag.origin.y + screen.y - drag.start.y });
    } else if (drag.kind === "add") {
      setDrag({ ...drag, current: world });
    } else if (drag.kind === "endpoint") {
      setLines((current) => current.map((line) => {
        if (line.id !== drag.id) return line;
        const geometry = drag.endpoint === 1 ? { x1: world.x, y1: world.y } : { x2: world.x, y2: world.y };
        const next = { ...line, ...geometry, edited: true };
        const metrics = lineMetrics(next);
        return {
          ...next,
          width_px: metrics.width_px,
          width_nm: calibration == null ? null : metrics.width_px * calibration,
          angle_deg: metrics.angle_deg,
        };
      }));
    }
  }

  function onPointerUp() {
    if (drag?.kind === "add") {
      const metrics = lineMetrics({ x1: drag.start.x, y1: drag.start.y, x2: drag.current.x, y2: drag.current.y });
      if (metrics.width_px > 2) {
        const id = `local-${crypto.randomUUID()}`;
        const added: CanvasLine = {
          id,
          source_model_measurement_id: null,
          active: true,
          source: "manual",
          edited: false,
          confidence: null,
          x1: drag.start.x,
          y1: drag.start.y,
          x2: drag.current.x,
          y2: drag.current.y,
          width_px: metrics.width_px,
          width_nm: calibration == null ? null : metrics.width_px * calibration,
          angle_deg: metrics.angle_deg,
        };
        commitLines([...lines, added]);
        setSelectedId(id);
      }
    } else if (drag?.kind === "endpoint") {
      setPast((history) => [...history.slice(-49), drag.before]);
      setFuture([]);
    }
    setDrag(null);
  }

  function onWheel(event: ReactWheelEvent<HTMLCanvasElement>) {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const world = screenToWorld(point);
    const factor = Math.exp(-event.deltaY * 0.0015);
    const nextScale = Math.min(16, Math.max(0.04, scale * factor));
    setScale(nextScale);
    setOffset({ x: point.x - world.x * nextScale, y: point.y - world.y * nextScale });
  }

  function chooseSplit(value: string) {
    if (value === "whole") {
      setSplitCount(null);
      setSectorIndex(0);
      window.requestAnimationFrame(() => fitWhole());
      return;
    }
    const count = Number(value) as SplitCount;
    setSplitCount(count);
    setSectorIndex(0);
    window.requestAnimationFrame(() => fitCurrentSector(count, 0));
  }

  function moveSector(delta: number) {
    if (!splitCount) return;
    const next = (sectorIndex + delta + splitCount) % splitCount;
    setSectorIndex(next);
    fitCurrentSector(splitCount, next);
  }

  const selected = useMemo(() => lines.find((line) => line.id === selectedId && line.active) ?? null, [lines, selectedId]);
  const selectedMetrics = selected ? lineMetrics(selected) : null;
  const activeCount = lines.filter((line) => line.active).length;
  const lowConfidenceIndex = lowConfidenceLines.findIndex((line) => line.id === selectedId);

  return (
    <div className="review-workspace">
      <aside className="tool-rail" aria-label="검수 도구">
        <div className="tool-group">
          <span className="tool-group-label">편집</span>
          <ToolButton label="선택" glyph="↖" active={mode === "select"} onClick={() => setMode("select")}/>
          <ToolButton label="추가" glyph="＋" disabled={disabled} active={mode === "add"} onClick={() => setMode("add")}/>
          <ToolButton label="수정" glyph="↔" disabled={disabled} active={mode === "edit"} onClick={() => setMode("edit")}/>
          <ToolButton label="삭제" glyph="−" disabled={disabled} active={mode === "delete"} onClick={() => setMode("delete")}/>
        </div>
        <div className="tool-group">
          <span className="tool-group-label">이력</span>
          <ToolButton label="실행 취소" glyph="↶" disabled={disabled || !past.length} onClick={undo}/>
          <ToolButton label="다시 실행" glyph="↷" disabled={disabled || !future.length} onClick={redo}/>
        </div>
        <div className="tool-group">
          <span className="tool-group-label">보기</span>
          <ToolButton label="돋보기" glyph="⌕" active={magnifierOn} onClick={() => setMagnifierOn((value) => !value)}/>
          <ToolButton label="측정선" glyph="━" active={overlayOn} onClick={() => setOverlayOn((value) => !value)}/>
          <ToolButton label="맞춤" glyph="□" onClick={() => fitCurrentSector()}/>
        </div>
      </aside>

      <section className="viewer-panel">
        <div className="viewer-bar">
          <div className="viewer-bar-left">
            <strong>{activeCount.toLocaleString()} measurements</strong>
            <span className="legend"><i className="legend-auto"/>자동</span>
            <span className="legend"><i className="legend-manual"/>수동</span>
          </div>
          <div className="viewer-controls">
            <label className="compact-select">필터
              <select value={lowConfidenceOnly ? "low" : "all"} onChange={(event) => {
                const lowOnly = event.target.value === "low";
                setLowConfidenceOnly(lowOnly);
                if (lowOnly && lowConfidenceLines.length) window.requestAnimationFrame(() => moveLowConfidence(1, true));
              }}>
                <option value="all">전체</option>
                <option value="low">낮은 신뢰도 &lt; 0.70</option>
              </select>
            </label>
            <label className="compact-select">분할
              <select value={splitCount ?? "whole"} onChange={(event) => chooseSplit(event.target.value)}>
                <option value="whole">전체</option>
                {SPLITS.map((count) => <option key={count} value={count}>{count}분할</option>)}
              </select>
            </label>
            {splitCount && (
              <div className="sector-nav">
                <button onClick={() => moveSector(-1)} aria-label="이전 영역">‹</button>
                <strong>{sectorIndex + 1} / {splitCount}</strong>
                <button onClick={() => moveSector(1)} aria-label="다음 영역">›</button>
              </div>
            )}
            {lowConfidenceOnly && (
              <div className="confidence-nav" title="낮은 신뢰도 측정선 순차 검수">
                <button disabled={!lowConfidenceLines.length} onClick={() => moveLowConfidence(-1)} aria-label="이전 낮은 신뢰도 측정선">‹</button>
                <strong>{lowConfidenceLines.length ? `${lowConfidenceIndex >= 0 ? lowConfidenceIndex + 1 : 0} / ${lowConfidenceLines.length}` : "0 / 0"}</strong>
                <button disabled={!lowConfidenceLines.length} onClick={() => moveLowConfidence(1)} aria-label="다음 낮은 신뢰도 측정선">›</button>
              </div>
            )}
          </div>
        </div>

        <div ref={shellRef} className="canvas-shell">
          <canvas
            ref={canvasRef}
            className={`canvas mode-${mode}`}
            tabIndex={0}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onPointerLeave={() => setHoverPoint(null)}
            onWheel={onWheel}
          />
        </div>
        <div className="viewer-hint">
          <span>휠 확대/축소</span><span>빈 공간 드래그 이동</span><span>{mode === "edit" ? "흰 점을 드래그해 수정" : "선을 클릭해 선택"}</span><span>Ctrl+Z / Ctrl+Y 실행 취소·다시 실행</span>
        </div>
      </section>

      <aside className="inspector">
        <div className="inspector-title"><span>측정 정보</span>{selected && <span className="status-dot"/>}</div>
        {selected && selectedMetrics ? (
          <div className="inspector-content">
            <div className="metric-primary">
              <span>두께</span>
              <strong>{selectedMetrics.width_px.toFixed(2)} <small>px</small></strong>
              {calibration != null && <em>{(selectedMetrics.width_px * calibration).toFixed(2)} nm</em>}
            </div>
            <dl className="metric-list">
              <div><dt>Fiber 방향</dt><dd>{fiberAngleFromMeasurement(selectedMetrics.angle_deg).toFixed(1)}°</dd></div>
              <div><dt>구분</dt><dd>{sourceLabel(selected)}</dd></div>
              <div><dt>신뢰도</dt><dd>{selected.confidence == null ? "—" : `${(selected.confidence * 100).toFixed(1)}%`}</dd></div>
              <div><dt>상태</dt><dd>{selected.edited ? "수정됨" : "유지"}</dd></div>
            </dl>
            {!disabled && <button className="inspector-delete" onClick={() => removeLine(selected.id)}>측정선 삭제</button>}
            <p className="inspector-note">수정 모드에서 끝점의 흰 점을 드래그해 위치를 조정할 수 있습니다.</p>
          </div>
        ) : (
          <div className="inspector-empty"><span>+</span><p>측정선을 선택하면<br/>두께와 방향을 확인할 수 있습니다.</p></div>
        )}
      </aside>
    </div>
  );
}
