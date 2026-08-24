"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { createAnalysis, uploadImage } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [nmPerPixel, setNmPerPixel] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setBusy(true); setError(null);
    try {
      const image = await uploadImage(file, nmPerPixel ? Number(nmPerPixel) : undefined);
      const job = await createAnalysis(image.id);
      router.push(`/analysis/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err)); setBusy(false);
    }
  }

  return (
    <main className="page"><section className="hero">
      <span className="badge">SEM · Human in the loop</span>
      <h1>Fiber Thickness Analysis</h1>
      <p className="muted">SEM 이미지를 올리면 v6.11 모델과 wide-fiber recovery가 측정선을 생성합니다. 결과를 직접 수정한 뒤 승인하면 다음 학습 데이터로 누적됩니다.</p>
      <form className="card upload-box" onSubmit={submit}>
        <label className="file-input"><strong>SEM 이미지</strong><br/><input type="file" accept="image/jpeg,image/png,image/tiff,image/bmp" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></label>
        <label className="field"><span>nm / pixel <span className="muted">(알고 있으면 입력)</span></span><input type="number" min="0" step="any" value={nmPerPixel} onChange={(e) => setNmPerPixel(e.target.value)} placeholder="예: 2.0" /></label>
        <div className="actions"><button className="primary" disabled={!file || busy}>{busy ? "업로드 중…" : "분석 시작"}</button>{file && <span className="muted">{file.name}</span>}</div>
        {error && <p className="error">{error}</p>}
      </form>
    </section></main>
  );
}
