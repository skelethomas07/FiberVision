"use client";

import { useMemo, useState } from "react";
import { exportUrl } from "@/lib/api";

const exports = [
  { kind: "csv" as const, title: "측정 데이터 CSV", description: "좌표, 두께, 방향, 상태를 측정선별로 저장", extension: ".csv" },
  { kind: "orientation" as const, title: "배향 분포 CSV", description: "0°~180°를 1° 간격 181개 열로 저장 · 각 셀은 개수/전체 (비율)", extension: ".csv" },
  { kind: "overlay" as const, title: "측정 이미지", description: "최종 측정선이 표시된 SEM 이미지", extension: ".png" },
  { kind: "labeled" as const, title: "라벨 이미지", description: "측정 데이터 CSV 번호와 동일한 번호가 표시된 이미지", extension: ".png" },
];

export default function ExportPanel({ analysisId }: { analysisId: string }) {
  const [minNm, setMinNm] = useState("");
  const [maxNm, setMaxNm] = useState("");
  const range = useMemo(() => {
    if (!minNm.trim() || !maxNm.trim()) return null;
    const min = Number(minNm);
    const max = Number(maxNm);
    if (!Number.isFinite(min) || !Number.isFinite(max) || min < 0 || max < min) return null;
    return { normalMinNm: min, normalMaxNm: max };
  }, [minNm, maxNm]);

  return (
    <section className="export-panel">
      <div className="section-heading-row">
        <div><p className="eyebrow">EXPORT</p><h2>내보내기</h2><p>저장된 최종 검수 결과만 파일에 포함됩니다.</p></div>
      </div>

      <div className="export-list">
        {exports.map((item) => (
          <article key={item.kind} className="export-item">
            <div className="file-badge">{item.extension}</div>
            <div className="export-copy"><strong>{item.title}</strong><span>{item.description}</span></div>
            <a className="download-button" href={exportUrl(analysisId, item.kind)}>다운로드 ↓</a>
          </article>
        ))}

        <article className="export-item" style={{ gridTemplateColumns: "48px minmax(0, 1fr) auto" }}>
          <div className="file-badge">.csv</div>
          <div className="export-copy" style={{ gap: 10 }}>
            <div><strong>두께 정상범위 CSV</strong><span style={{ display: "block", marginTop: 4 }}>지정한 nm 범위에 들어오는 Fiber 수 / 전체 Fiber 수 / 비율을 저장</span></div>
            <div style={{ display: "flex", alignItems: "end", gap: 8, maxWidth: 330 }}>
              <label className="compact-field" style={{ flex: 1 }}><span>최솟값 (nm)</span><input type="number" min="0" step="0.1" value={minNm} onChange={(e) => setMinNm(e.target.value)} placeholder="예: 40"/></label>
              <span style={{ paddingBottom: 9, color: "#7b8797" }}>~</span>
              <label className="compact-field" style={{ flex: 1 }}><span>최댓값 (nm)</span><input type="number" min="0" step="0.1" value={maxNm} onChange={(e) => setMaxNm(e.target.value)} placeholder="예: 80"/></label>
            </div>
          </div>
          {range
            ? <a className="download-button" href={exportUrl(analysisId, "thickness-range", range)}>다운로드 ↓</a>
            : <button className="download-button" disabled>범위 입력</button>}
        </article>
      </div>

      <article className="bundle-card">
        <div><span className="bundle-icon" aria-hidden="true">▣</span><div><strong>기본 전체 파일</strong><p>측정 데이터 CSV, 측정 이미지, 라벨 이미지를 ZIP 하나로 받습니다.</p></div></div>
        <a className="primary download-button" href={exportUrl(analysisId, "bundle")}>전체 ZIP 다운로드</a>
      </article>
    </section>
  );
}
