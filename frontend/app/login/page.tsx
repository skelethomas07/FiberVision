"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, getCurrentUser, login } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCurrentUser().then((user) => {
      router.replace(user.must_change_password ? "/change-password" : "/");
    }).catch(() => undefined);
  }, [router]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const user = await login(email, password);
      router.replace(user.must_change_password ? "/change-password" : "/");
    } catch (err) {
      if (err instanceof ApiError && err.code === "INVALID_CREDENTIALS") setError("이메일 또는 비밀번호를 확인해 주세요.");
      else setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-card">
        <div className="brand-mark" aria-hidden="true"><span/><span/><span/></div>
        <div className="auth-heading">
          <p className="eyebrow">SEM ANALYSIS WORKSPACE</p>
          <h1>FiberVision</h1>
          <p>등록된 계정으로 로그인하세요.</p>
        </div>
        <form className="auth-form" onSubmit={submit}>
          <label><span>이메일</span><input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@example.com"/></label>
          <label><span>비밀번호</span><input type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} placeholder="비밀번호"/></label>
          {error && <p className="form-error">{error}</p>}
          <button className="primary auth-submit" disabled={busy}>{busy ? "로그인 중…" : "로그인"}</button>
        </form>
      </section>
    </main>
  );
}
