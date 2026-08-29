"use client";

import { useMemo, useState } from "react";
import { directionBins, fiberAngleFromMeasurement, histogramBins, measurementStats } from "@/lib/reviewView";
import type { ReviewMeasurement } from "@/lib/api";

type Filter = "all" | "auto" | "manual" | "corrected";

function matchesFilter(line: ReviewMeasurement, filter: Filter) {
  if (!line.active) return false;
  if (filter === "all") return true;
  if (filter === "corrected") return line.edited;
  const manual = line.source_model_measurement_id === null || line.source === "manual" || line.source === "visionflux_manual";
  if (filter === "manual") return manual;
  return !manual && !line.edited;
}

function format(value: number, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function Histogram({ values, unit }: { values: number[]; unit: string }) {
  const bins = histogramBins(values, 14);
  const maxCount = Math.max(1, ...bins.map((bin) => bin.count));
  if (!bins.length) return <div className="chart-empty">표시할 측정값이 없습니다.</div>;
  return (
    <div className="chart-wrap">
      <svg className="histogram" viewBox="0 0 720 260" role="img" aria-label="두께 분포 히스토그램">
        <line x1="40" y1="220" x2="700" y2="220" className="chart-axis"/>
        {bins.map((bin, index) => {
          const gap = 5;
          const available = 660 / bins.length;
          const height = (bin.count / maxCount) * 180;
          return <rect key={index} x={40 + index * available + gap / 2} y={220 - height} width={Math.max(2, available - gap)} height={height} rx="3" className="chart-bar"/>;
        })}
        <text x="40" y="246" className="chart-label">{format(bins[0].start)} {unit}</text>
        <text x="700" y="246" textAnchor="end" className="chart-label">{format(bins[bins.length - 1].end)} {unit}</text>
      </svg>
    </div>
  );
}

function RoseChart({ angles }: { angles: number[] }) {
  const bins = directionBins(angles, 12);
  const maxCount = Math.max(1, ...bins.map((bin) => bin.count));
  const center = 140;
  const radius = 96;
  if (!angles.length) return <div className="chart-empty">표시할 방향 데이터가 없습니다.</div>;
  return (
    <div className="rose-wrap">
      <svg className="rose-chart" viewBox="0 0 280 280" role="img" aria-label="Fiber 방향 분포">
        {[0.33, 0.66, 1].map((fraction) => <circle key={fraction} cx={center} cy={center} r={radius * fraction} className="rose-grid"/>)}
        <line x1={center - radius} y1={center} x2={center + radius} y2={center} className="rose-grid"/>
        <line x1={center} y1={center - radius} x2={center} y2={center + radius} className="rose-grid"/>
        {bins.map((bin, index) => {
          const angle = ((bin.start + bin.end) / 2 - 90) * Math.PI / 180;
          const length = radius * (bin.count / maxCount);
          return <line key={index} x1={center} y1={center} x2={center + Math.cos(angle) * length} y2={center + Math.sin(angle) * length} className="rose-bar"/>;
        })}
        <text x={center} y="24" textAnchor="middle" className="chart-label">90°</text>
        <text x="258" y={center + 4} textAnchor="end" className="chart-label">0° / 180°</text>
      </svg>
    </div>
  );
}

export default function ResultsPanel({ measurements }: { measurements: ReviewMeasurement[] }) {
  const [filter, setFilter] = useState<Filter>("all");
  const filtered = useMemo(() => measurements.filter((line) => matchesFilter(line, filter)), [filter, measurements]);
  const useNm = filtered.length > 0 && filtered.every((line) => line.width_nm != null);
  const unit = useNm ? "nm" : "px";
  const thickness = filtered.map((line) => useNm ? Number(line.width_nm) : line.width_px);
  const stats = measurementStats(thickness);
  const angles = filtered.map((line) => fiberAngleFromMeasurement(line.angle_deg));

  return (
    <section className="results-panel">
      <div className="section-heading-row">
        <div><p className="eyebrow">MEASUREMENT SUMMARY</p><h2>결과</h2><p>검수에서 저장된 활성 측정값을 기준으로 계산합니다.</p></div>
        <div className="segmented" aria-label="결과 필터">
          {([ ["all", "전체"], ["auto", "자동"], ["manual", "수동"], ["corrected", "수정"] ] as Array<[Filter, string]>).map(([value, label]) => (
            <button key={value} className={filter === value ? "active" : ""} onClick={() => setFilter(value)}>{label}</button>
          ))}
        </div>
      </div>

      <div className="stat-grid">
        <article><span>측정 수</span><strong>{stats.count.toLocaleString()}</strong><small>measurements</small></article>
        <article><span>평균 두께</span><strong>{format(stats.mean)}</strong><small>{unit}</small></article>
        <article><span>중앙값</span><strong>{format(stats.median)}</strong><small>{unit}</small></article>
        <article><span>표준편차</span><strong>{format(stats.stddev)}</strong><small>{unit}</small></article>
      </div>

      <div className="chart-grid">
        <article className="result-card">
          <div className="card-title"><div><span>Thickness</span><h3>두께 분포</h3></div><em>{unit}</em></div>
          <Histogram values={thickness} unit={unit}/>
        </article>
        <article className="result-card">
          <div className="card-title"><div><span>Orientation</span><h3>방향 분포</h3></div><em>0–180°</em></div>
          <RoseChart angles={angles}/>
        </article>
      </div>
    </section>
  );
}
