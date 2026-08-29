"use client";

import { exportUrl } from "@/lib/api";

const exports = [
  { kind: "csv" as const, title: "CSV", description: "좌표, 두께, 방향, 상태를 표 형식으로 저장", extension: ".csv" },
  { kind: "overlay" as const, title: "측정 이미지", description: "최종 측정선이 표시된 SEM 이미지", extension: ".png" },
  { kind: "labeled" as const, title: "라벨 이미지", description: "CSV 번호와 동일한 번호가 표시된 이미지", extension: ".png" },
];

export default function ExportPanel({ analysisId }: { analysisId: string }) {
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
      </div>
      <article className="bundle-card">
        <div><span className="bundle-icon" aria-hidden="true">▣</span><div><strong>전체 파일</strong><p>CSV, 측정 이미지, 라벨 이미지를 ZIP 하나로 받습니다.</p></div></div>
        <a className="primary download-button" href={exportUrl(analysisId, "bundle")}>전체 ZIP 다운로드</a>
      </article>
    </section>
  );
}
