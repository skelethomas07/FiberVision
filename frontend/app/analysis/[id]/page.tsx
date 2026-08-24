"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import AnalysisCanvas from "@/components/MeasurementCanvas";
import { AnalysisStatus } from "@/components/AnalysisStatus";
import { absoluteImageUrl, approveReview, getAnalysis, getResult, getReview, saveReview, type AnalysisJob, type AnalysisResult, type ReviewResponse } from "@/lib/api";
import type { ReviewPatch } from "@/lib/geometry";

const EMPTY_PATCH: ReviewPatch = { removed_ids: [], corrected: [], added: [] };

export default function AnalysisPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [job, setJob] = useState<AnalysisJob | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [review, setReview] = useState<ReviewResponse | null>(null);
  const [patch, setPatch] = useState<ReviewPatch>(EMPTY_PATCH);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    async function poll() {
      try {
        const next = await getAnalysis(id);
        if (stopped) return;
        setJob(next);
        if (next.status === "DONE") {
          const [r, rv] = await Promise.all([getResult(id), getReview(id)]);
          if (!stopped) { setResult(r); setReview(rv); }
          return;
        }
        if (next.status === "FAILED") return;
        window.setTimeout(poll, 1500);
      } catch (err) { if (!stopped) setError(err instanceof Error ? err.message : String(err)); }
    }
    poll();
    return () => { stopped = true; };
  }, [id]);

  const onPatchChange = useCallback((next: ReviewPatch) => setPatch(next), []);
  const changeCount = patch.removed_ids.length + patch.corrected.length + patch.added.length;

  async function save() {
    if (!review || review.status !== "OPEN") return review;
    setSaving(true); setError(null);
    try {
      const next = await saveReview(review.id, patch);
      setReview(next); setPatch(EMPTY_PATCH); setMessage("검수 수정사항을 저장했습니다.");
      return next;
    } finally { setSaving(false); }
  }

  async function approve() {
    if (!review) return;
    setSaving(true); setError(null); setMessage(null);
    try {
      let current = review;
      if (changeCount) current = await saveReview(review.id, patch);
      const approved = await approveReview(current.id);
      setReview({ ...current, status: "APPROVED" });
      setPatch(EMPTY_PATCH);
      setMessage(`검수 완료 · ${approved.training_examples}개 supervision이 학습 데이터베이스에 추가되었습니다.`);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setSaving(false); }
  }

  if (error && !job) return <main className="page"><p className="error">{error}</p></main>;
  if (!job || job.status !== "DONE" || !result || !review) return <main className="page">{job ? <AnalysisStatus job={job}/> : <p>분석 정보를 불러오는 중…</p>}</main>;

  const active = review.measurements.filter((m) => m.active).length;
  return (
    <main className="page">
      <div className="topbar"><div><span className="badge">Analysis {id.slice(0, 8)}</span><h2>Fiber 측정 결과 검수</h2><div className="summary"><span>{active} active measurements</span><span>{String(result.summary["units"] ?? "px")}</span><span>{job.model_version}</span></div></div><a href="/">새 이미지 분석</a></div>
      <AnalysisCanvas imageUrl={absoluteImageUrl(result.image_url)} measurements={review.measurements} disabled={review.status === "APPROVED"} onPatchChange={onPatchChange}/>
      <div className="review-actions">
        <span className="muted">변경 {changeCount}건</span>
        <button disabled={saving || !changeCount || review.status === "APPROVED"} onClick={save}>수정사항 저장</button>
        <button className="primary" disabled={saving || review.status === "APPROVED"} onClick={approve}>검수 완료 · 학습 데이터 추가</button>
      </div>
      {message && <p className="success">{message}</p>}{error && <p className="error">{error}</p>}
    </main>
  );
}
