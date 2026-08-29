"use client";

import { DragEvent, FormEvent, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import AuthGuard from "@/components/AuthGuard";
import UserMenu from "@/components/UserMenu";
import { createAnalysis, uploadImage, type AuthUser } from "@/lib/api";

export default function HomePage() {
  return <AuthGuard>{(user) => <UploadWorkspace user={user}/>}</AuthGuard>;
}

function UploadWorkspace({ user }: { user: AuthUser }) {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [nmPerPixel, setNmPerPixel] = useState("");
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function acceptFile(next: File | null) {
    if (!next) return;
    setFile(next);
    setError(null);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const image = await uploadImage(file, nmPerPixel ? Number(nmPerPixel) : undefined);
      const job = await createAnalysis(image.id);
      router.push(`/analysis/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  return (
    <main className="app-shell home-shell">
      <header className="app-header">
        <a className="brand" href="/" aria-label="FiberVision home">
          <span className="brand-symbol" aria-hidden="true"><i/><i/><i/></span>
          <span>FiberVision</span>
        </a>
        <UserMenu user={user}/>
      </header>

      <section className="home-content">
        <div className="home-intro">
          <p className="eyebrow">SEM FIBER ANALYSIS</p>
          <h1><span>이미지에서 측정까지.</span><span>검수는 필요한 곳만.</span></h1>
          <p>원본 SEM과 VisionFlux 결과 이미지를 같은 곳에서 분석하고 이어서 검수할 수 있습니다.</p>
        </div>

        <form className="upload-card" onSubmit={submit}>
          <div
            className={`drop-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }}
          >
            <input
              ref={inputRef}
              className="sr-only"
              type="file"
              accept="image/jpeg,image/png,image/tiff,image/bmp,.tif,.tiff"
              onChange={(event) => acceptFile(event.target.files?.[0] ?? null)}
            />
            <div className="upload-icon" aria-hidden="true">↑</div>
            {file ? (
              <>
                <strong>{file.name}</strong>
                <span>{(file.size / 1024 / 1024).toFixed(2)} MB · 클릭해서 다른 파일 선택</span>
              </>
            ) : (
              <>
                <strong>SEM 이미지를 놓아주세요</strong>
                <span>또는 클릭해서 파일 선택 · JPG, PNG, TIFF</span>
              </>
            )}
          </div>

          <div className="upload-options">
            <label className="compact-field">
              <span>nm / pixel</span>
              <input type="number" min="0" step="any" value={nmPerPixel} onChange={(event) => setNmPerPixel(event.target.value)} placeholder="선택사항"/>
            </label>
            <p>VisionFlux 노란색·파란색 측정선이 포함된 이미지도 자동으로 인식합니다.</p>
          </div>

          {error && <p className="form-error">{error}</p>}
          <button className="primary upload-submit" disabled={!file || busy}>{busy ? "분석 준비 중…" : "분석 시작"}</button>
        </form>
      </section>
    </main>
  );
}
