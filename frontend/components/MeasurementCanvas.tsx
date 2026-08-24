"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildReviewPatch, lineMetrics, type ReviewLine, type ReviewPatch } from "@/lib/geometry";

type CanvasLine = ReviewLine & { source: string; edited?: boolean };
type Props = {
  imageUrl: string;
  measurements: CanvasLine[];
  disabled?: boolean;
  onPatchChange: (patch: ReviewPatch) => void;
};

type Point = { x: number; y: number };
type DragState =
  | { kind: "pan"; start: Point; origin: Point }
  | { kind: "endpoint"; id: string; endpoint: 1 | 2 }
  | { kind: "add"; start: Point; current: Point }
  | null;

const LENS_RADIUS = 92;
const LENS_ZOOM = 2.5;

function segmentDistance(p: Point, line: CanvasLine) {
  const vx = line.x2 - line.x1, vy = line.y2 - line.y1;
  const wx = p.x - line.x1, wy = p.y - line.y1;
  const len2 = vx * vx + vy * vy;
  const t = len2 ? Math.max(0, Math.min(1, (wx * vx + wy * vy) / len2)) : 0;
  return Math.hypot(p.x - (line.x1 + t * vx), p.y - (line.y1 + t * vy));
}

function lineColor(line: CanvasLine, selected: boolean) {
  if (selected) return "#ff4d6d";
  if (line.source === "manual" || line.source === "visionflux_manual") return "#4aa3ff";
  if (line.source === "thick_recovery") return "#ff735c";
  return "#ffd34d";
}

export default function MeasurementCanvas({ imageUrl, measurements, disabled = false, onPatchChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const initialRef = useRef<CanvasLine[]>(measurements.map((m) => ({ ...m })));
  const [lines, setLines] = useState<CanvasLine[]>(measurements.map((m) => ({ ...m })));
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [addMode, setAddMode] = useState(false);
  const [deleteMode, setDeleteMode] = useState(false);
  const [magnifierOn, setMagnifierOn] = useState(true);
  const [hoverPoint, setHoverPoint] = useState<Point | null>(null);
  const [drag, setDrag] = useState<DragState>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 });
  const [size, setSize] = useState({ width: 800, height: 600 });

  useEffect(() => {
    initialRef.current = measurements.map((m) => ({ ...m }));
    setLines(measurements.map((m) => ({ ...m })));
    setSelectedId(null);
  }, [measurements]);

  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) return;
    const ro = new ResizeObserver(([entry]) => setSize({ width: entry.contentRect.width, height: Math.max(420, entry.contentRect.height) }));
    ro.observe(shell);
    return () => ro.disconnect();
  }, []);

  const fit = useCallback(() => {
    const image = imageRef.current;
    if (!image || !image.naturalWidth || !image.naturalHeight) return;
    const s = Math.min(size.width / image.naturalWidth, size.height / image.naturalHeight) * 0.94;
    setScale(s);
    setOffset({ x: (size.width - image.naturalWidth * s) / 2, y: (size.height - image.naturalHeight * s) / 2 });
  }, [size]);

  useEffect(() => {
    const image = new Image();
    image.onload = () => { imageRef.current = image; fit(); };
    image.onerror = () => { imageRef.current = null; };
    image.src = imageUrl;
    return () => { imageRef.current = null; };
  }, [imageUrl, fit]);

  useEffect(() => { fit(); }, [fit]);
  useEffect(() => { onPatchChange(buildReviewPatch(initialRef.current ?? [], lines)); }, [lines, onPatchChange]);

  const screenToWorld = useCallback((p: Point) => ({ x: (p.x - offset.x) / scale, y: (p.y - offset.y) / scale }), [offset, scale]);
  const worldToScreen = useCallback((p: Point) => ({ x: offset.x + p.x * scale, y: offset.y + p.y * scale }), [offset, scale]);

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
    ctx.fillStyle = "#0b1020";
    ctx.fillRect(0, 0, size.width, size.height);
    ctx.drawImage(image, offset.x, offset.y, image.naturalWidth * scale, image.naturalHeight * scale);

    const drawLines = (mapper: (p: Point) => Point, selectedHandles = true) => {
      for (const line of lines) {
        if (!line.active) continue;
        const a = mapper({ x: line.x1, y: line.y1 });
        const b = mapper({ x: line.x2, y: line.y2 });
        const selected = line.id === selectedId;
        ctx.strokeStyle = lineColor(line, selected);
        ctx.lineWidth = selected ? 3 : 1.7;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        if (selected && selectedHandles) {
          ctx.fillStyle = "#ffffff";
          for (const p of [a, b]) { ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill(); }
        }
      }
    };
    drawLines(worldToScreen);

    if (drag?.kind === "add") {
      const a = worldToScreen(drag.start), b = worldToScreen(drag.current);
      ctx.strokeStyle = "#4aa3ff"; ctx.lineWidth = 2; ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.setLineDash([]);
    }

    if (magnifierOn && hoverPoint && !drag) {
      const world = screenToWorld(hoverPoint);
      const lensMap = (p: Point) => ({
        x: hoverPoint.x + (p.x - world.x) * scale * LENS_ZOOM,
        y: hoverPoint.y + (p.y - world.y) * scale * LENS_ZOOM,
      });
      ctx.save();
      ctx.beginPath(); ctx.arc(hoverPoint.x, hoverPoint.y, LENS_RADIUS, 0, Math.PI * 2); ctx.clip();
      ctx.fillStyle = "#0b1020"; ctx.fillRect(hoverPoint.x - LENS_RADIUS, hoverPoint.y - LENS_RADIUS, LENS_RADIUS * 2, LENS_RADIUS * 2);
      ctx.drawImage(
        image,
        hoverPoint.x - world.x * scale * LENS_ZOOM,
        hoverPoint.y - world.y * scale * LENS_ZOOM,
        image.naturalWidth * scale * LENS_ZOOM,
        image.naturalHeight * scale * LENS_ZOOM,
      );
      drawLines(lensMap, false);
      ctx.restore();
      ctx.beginPath(); ctx.arc(hoverPoint.x, hoverPoint.y, LENS_RADIUS, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,.92)"; ctx.lineWidth = 2; ctx.stroke();
    }
  }, [size, lines, selectedId, drag, offset, scale, worldToScreen, screenToWorld, magnifierOn, hoverPoint]);

  const pointer = (event: React.PointerEvent<HTMLCanvasElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const nearestLine = useCallback((screen: Point) => {
    const world = screenToWorld(screen);
    return lines
      .filter((line) => line.active)
      .map((line) => ({ line, d: segmentDistance(world, line) * scale }))
      .sort((a, b) => a.d - b.d)[0];
  }, [lines, scale, screenToWorld]);

  const removeLine = useCallback((id: string) => {
    if (disabled) return;
    setLines((prev) => prev.flatMap((line) => line.id !== id ? [line] : line.id.startsWith("local-") ? [] : [{ ...line, active: false }]));
    setSelectedId((current) => current === id ? null : current);
  }, [disabled]);

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    const s = pointer(event), w = screenToWorld(s);
    if (!disabled && deleteMode) {
      const hit = nearestLine(s);
      if (hit && hit.d < 11) removeLine(hit.line.id);
      return;
    }
    if (!disabled && addMode) { setDrag({ kind: "add", start: w, current: w }); return; }

    if (!disabled && selectedId) {
      const selected = lines.find((line) => line.id === selectedId && line.active);
      if (selected) {
        const a = worldToScreen({ x: selected.x1, y: selected.y1 }), b = worldToScreen({ x: selected.x2, y: selected.y2 });
        if (Math.hypot(s.x - a.x, s.y - a.y) < 12) { setDrag({ kind: "endpoint", id: selected.id, endpoint: 1 }); return; }
        if (Math.hypot(s.x - b.x, s.y - b.y) < 12) { setDrag({ kind: "endpoint", id: selected.id, endpoint: 2 }); return; }
      }
    }

    const hit = nearestLine(s);
    if (hit && hit.d < 9) { setSelectedId(hit.line.id); return; }
    setSelectedId(null);
    setDrag({ kind: "pan", start: s, origin: offset });
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const s = pointer(event);
    setHoverPoint(s);
    if (!drag) return;
    const w = screenToWorld(s);
    if (drag.kind === "pan") setOffset({ x: drag.origin.x + s.x - drag.start.x, y: drag.origin.y + s.y - drag.start.y });
    if (drag.kind === "add") setDrag({ ...drag, current: w });
    if (drag.kind === "endpoint") setLines((prev) => prev.map((line) => line.id !== drag.id ? line : ({ ...line, ...(drag.endpoint === 1 ? { x1: w.x, y1: w.y } : { x2: w.x, y2: w.y }), edited: true })));
  };

  const onPointerUp = () => {
    if (drag?.kind === "add") {
      const metrics = lineMetrics({ x1: drag.start.x, y1: drag.start.y, x2: drag.current.x, y2: drag.current.y });
      if (metrics.width_px > 2) {
        const id = `local-${crypto.randomUUID()}`;
        setLines((prev) => [...prev, { id, source_model_measurement_id: null, active: true, source: "manual", x1: drag.start.x, y1: drag.start.y, x2: drag.current.x, y2: drag.current.y }]);
        setSelectedId(id);
      }
    }
    setDrag(null);
  };

  const onWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const p = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const world = screenToWorld(p);
    const factor = Math.exp(-event.deltaY * 0.0015);
    const next = Math.min(12, Math.max(0.05, scale * factor));
    setScale(next);
    setOffset({ x: p.x - world.x * next, y: p.y - world.y * next });
  };

  const selected = useMemo(() => lines.find((line) => line.id === selectedId), [lines, selectedId]);
  const activeCount = lines.filter((line) => line.active).length;

  return (
    <div className="viewer">
      <div className="viewer-toolbar">
        <button className={addMode ? "active" : ""} disabled={disabled} onClick={() => { setAddMode((v) => !v); setDeleteMode(false); }}>+ 측정선 추가</button>
        <button className={deleteMode ? "danger-active" : ""} disabled={disabled} onClick={() => { setDeleteMode((v) => !v); setAddMode(false); }}>측정선 삭제</button>
        <button className={magnifierOn ? "active" : ""} onClick={() => setMagnifierOn((v) => !v)}>돋보기 {magnifierOn ? "ON" : "OFF"}</button>
        <button onClick={fit}>화면 맞춤</button>
        <span>{activeCount} measurements</span>
        {selected && selected.active && <span className="selected-info">선택 {lineMetrics(selected).width_px.toFixed(1)} px</span>}
      </div>
      <div ref={shellRef} className="canvas-shell">
        <canvas
          ref={canvasRef}
          className={deleteMode ? "delete-mode" : addMode ? "add-mode" : ""}
          tabIndex={0}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onPointerLeave={() => setHoverPoint(null)}
          onWheel={onWheel}
        />
      </div>
      <p className="viewer-help">휠: 확대/축소 · 빈 공간 드래그: 이동 · 선 클릭: 선택 · 흰 점 드래그: 끝점 수정 · 측정선 삭제: 선을 클릭해 제거</p>
    </div>
  );
}
