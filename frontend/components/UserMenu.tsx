"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { logout, type AuthUser } from "@/lib/api";

export default function UserMenu({ user }: { user: AuthUser }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  async function signOut() {
    setBusy(true);
    try { await logout(); } finally { router.replace("/login"); }
  }

  return (
    <div className="user-menu">
      <button className="user-trigger" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span className="user-avatar">{user.email.slice(0, 1).toUpperCase()}</span>
        <span className="user-email">{user.email}</span>
        <span aria-hidden="true">⌄</span>
      </button>
      {open && (
        <div className="user-popover">
          <span className="user-popover-label">로그인 계정</span>
          <strong>{user.email}</strong>
          <button disabled={busy} onClick={signOut}>{busy ? "로그아웃 중…" : "로그아웃"}</button>
        </div>
      )}
    </div>
  );
}
