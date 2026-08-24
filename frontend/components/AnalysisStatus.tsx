import type { AnalysisJob } from "@/lib/api";

export function AnalysisStatus({ job }: { job: AnalysisJob }) {
  const label = {
    QUEUED: "분석 대기 중",
    ANALYZING: "Fiber 분석 중",
    POSTPROCESSING: "측정값 정리 중",
    DONE: "분석 완료",
    FAILED: "분석 실패",
  }[job.status];
  return (
    <section className="status-card">
      <div className="status-row"><strong>{label}</strong><span>{job.progress}%</span></div>
      <div className="progress"><div style={{ width: `${Math.max(3, job.progress)}%` }} /></div>
      <p className="muted">Model {job.model_version}</p>
      {job.error_message && <p className="error">{job.error_message}</p>}
    </section>
  );
}
