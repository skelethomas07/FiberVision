"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import AuthGuard from "@/components/AuthGuard";
import UserMenu from "@/components/UserMenu";
import MeasurementCanvas from "@/components/MeasurementCanvas";
import { AnalysisStatus } from "@/components/AnalysisStatus";
import ResultsPanel from "@/components/ResultsPanel";
import ExportPanel from "@/components/ExportPanel";
import {
  absoluteImageUrl,
  approveReview,
  getAnalysis,
  getResult,
  getReview,
  saveReview,
  type AnalysisJob,
  type AnalysisResult,
  type AuthUser,
  type ReviewResponse,
} from "@/lib/api";
import type { ReviewPatch } from "@/lib/geometry";

const EMPTY_PATCH: ReviewPatch = { removed_ids: [], corrected: [], added: [] };
type Tab = "review" | "results" | "export";

export default function AnalysisPage() {
  return <AuthGuard>{(user) => <AnalysisWorkspace user={user}/>}</AuthGuard>;
}

function AnalysisWorkspace({ user }: { user: AuthUser }) {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [job, setJob] = useState<AnalysisJob | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [review, setReview] = useState<ReviewResponse | null>(null);
  const [patch, setPatch] = useState<ReviewPatch>(EMPTY_PATCH);
  const [tab, setTab] = useState<Tab>("review");
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
          const [analysisResult, reviewResult] = await Promise.all([getResult(id), getReview(id)]);
          if (!stopped) {
            setResult(analysisResult);
            setReview(reviewResult);
          }
          return;
        }
        if (next.status === "FAILED") return;
        window.setTimeout(poll, 1500);
      } catch (err) {
        if (!stopped) setError(err instanceof Error ? err.message : String(err));
      }
    }
    poll();
    return () => { stopped = true; };
  }, [id]);

  const onPatchChange = useCallback((next: ReviewPatch) => setPatch(next), []);
  const changeCount = patch.removed_ids.length + patch.corrected.length + patch.added.length;

  useEffect(() => {
    if (!changeCount) return;
    const warnBeforeLeave = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warnBeforeLeave);
    return () => window.removeEventListener("beforeunload", warnBeforeLeave);
  }, [changeCount]);

  async function save(silent = false): Promise<ReviewResponse | null> {
    if (!review || review.status !== "OPEN") return review;
    if (!changeCount) return review;
    setSaving(true);
    setError(null);
    try {
      const next = await saveReview(review.id, patch);
      setReview(next);
      setPatch(EMPTY_PATCH);
      if (!silent) setMessage("수정사항을 저장했습니다.");
      return next;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally {
      setSaving(false);
    }
  }

  async function chooseTab(nextTab: Tab) {
    if (nextTab === tab) return;
    setMessage(null);
    if (tab === "review" && nextTab !== "review" && changeCount > 0) {
      const saved = await save(true);
      if (!saved) return;
    }
    setTab(nextTab);
  }

  async function approve() {
    if (!review) return;
    if (!window.confirm("검수를 완료하면 현재 결과가 확정되고 학습 데이터에 반영됩니다. 계속할까요?")) return;
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      let current = review;
      if (changeCount) current = await saveReview(review.id, patch);
      const approved = await approveReview(current.id);
      setReview({ ...current, status: "APPROVED" });
      setPatch(EMPTY_PATCH);
      setMessage(`검수 완료 · ${approved.training_examples.toLocaleString()}개 학습 데이터가 추가되었습니다.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  if (error && !job) return <main className="app-shell"><p className="error">{error}</p></main>;
  if (!job || job.status !== "DONE" || !result || !review) {
    return (
      <main className="app-shell analysis-shell">
        <header className="app-header">
          <a className="brand" href="/"><span className="brand-symbol" aria-hidden="true"><i/><i/><i/></span><span>FiberVision</span></a>
          <UserMenu user={user}/>
        </header>
        {job ? <AnalysisStatus job={job}/> : <div className="auth-loading"><span className="spinner"/>분석 정보를 불러오는 중</div>}
      </main>
    );
  }

  const active = review.measurements.filter((measurement) => measurement.active).length;

  return (
    <main className="app-shell analysis-shell">
      <header className="app-header analysis-header">
        <div className="analysis-brand-row">
          <a className="brand" href="/"><span className="brand-symbol" aria-hidden="true"><i/><i/><i/></span><span>FiberVision</span></a>
          <span className="header-divider"/>
          <div className="analysis-meta">
            <strong className="analysis-filename" title={result.image_filename}>{result.image_filename}</strong>
            <span>Analysis {id.slice(0, 8)} · {active.toLocaleString()} measurements · {job.model_version}</span>
          </div>
        </div>
        <div className="header-actions"><a className="text-action" href="/">새 분석</a><UserMenu user={user}/></div>
      </header>

      <nav className="analysis-tabs" aria-label="분석 메뉴">
        <button className={tab === "review" ? "active" : ""} onClick={() => chooseTab("review")}>검수</button>
        <button className={tab === "results" ? "active" : ""} onClick={() => chooseTab("results")}>결과</button>
        <button className={tab === "export" ? "active" : ""} onClick={() => chooseTab("export")}>내보내기</button>
        <div className="tab-status">
          <span className={`review-state ${review.status === "APPROVED" ? "approved" : ""}`}>{review.status === "APPROVED" ? "검수 완료" : "검수 중"}</span>
          {changeCount > 0 && <span>{changeCount}건 변경</span>}
        </div>
      </nav>

      <div className="analysis-body">
        {tab === "review" && (
          <MeasurementCanvas
            imageUrl={absoluteImageUrl(result.image_url)}
            measurements={review.measurements}
            disabled={review.status === "APPROVED"}
            onPatchChange={onPatchChange}
          />
        )}
        {tab === "results" && <ResultsPanel measurements={review.measurements}/>}
        {tab === "export" && <ExportPanel analysisId={id}/>}
      </div>

      {tab === "review" && (
        <div className="review-dock">
          <div>
            <span className="review-dock-label">현재 검수</span>
            <strong>{changeCount ? `${changeCount}건의 저장되지 않은 변경` : review.status === "APPROVED" ? "검수가 완료되었습니다" : "저장된 상태입니다"}</strong>
          </div>
          <div className="review-dock-actions">
            <button disabled={saving || !changeCount || review.status === "APPROVED"} onClick={() => save(false)}>{saving ? "저장 중…" : "저장"}</button>
            <button className="primary" disabled={saving || review.status === "APPROVED"} onClick={approve}>검수 완료</button>
          </div>
        </div>
      )}

      {(message || error) && <div className={`toast ${error ? "toast-error" : "toast-success"}`}>{error ?? message}</div>}
    </main>
  );
}
