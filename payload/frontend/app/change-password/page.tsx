"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import AuthGuard from "@/components/AuthGuard";
import { changePassword } from "@/lib/api";

export default function ChangePasswordPage() {
  return <AuthGuard allowPasswordChange>{(user) => <ChangePasswordForm email={user.email}/>}</AuthGuard>;
}

function ChangePasswordForm({ email }: { email: string }) {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    if (password !== confirmation) {
      setError("새 비밀번호가 서로 일치하지 않습니다.");
      return;
    }
    if (password.length < 10) {
      setError("비밀번호는 10자 이상 입력해 주세요.");
      return;
    }
    setBusy(true);
    try {
      await changePassword(password);
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-card">
        <div className="brand-mark" aria-hidden="true"><span/><span/><span/></div>
        <div className="auth-heading">
          <p className="eyebrow">FIRST SIGN IN</p>
          <h1>비밀번호 변경</h1>
          <p><strong>{email}</strong><br/>처음 로그인한 계정은 새 비밀번호를 설정해야 합니다.</p>
        </div>
        <form className="auth-form" onSubmit={submit}>
          <label><span>새 비밀번호</span><input type="password" autoComplete="new-password" required minLength={10} value={password} onChange={(e) => setPassword(e.target.value)}/></label>
          <label><span>비밀번호 확인</span><input type="password" autoComplete="new-password" required minLength={10} value={confirmation} onChange={(e) => setConfirmation(e.target.value)}/></label>
          {error && <p className="form-error">{error}</p>}
          <button className="primary auth-submit" disabled={busy}>{busy ? "변경 중…" : "비밀번호 변경"}</button>
        </form>
      </section>
    </main>
  );
}
