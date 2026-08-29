"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { ApiError, getCurrentUser, type AuthUser } from "@/lib/api";

type Props = {
  children: (user: AuthUser) => ReactNode;
  allowPasswordChange?: boolean;
};

export default function AuthGuard({ children, allowPasswordChange = false }: Props) {
  const router = useRouter();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getCurrentUser()
      .then((next) => {
        if (cancelled) return;
        if (next.must_change_password && !allowPasswordChange) {
          router.replace("/change-password");
          return;
        }
        setUser(next);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 401) {
          router.replace("/login");
          return;
        }
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => { cancelled = true; };
  }, [allowPasswordChange, router]);

  if (error) return <main className="auth-page"><div className="auth-card"><p className="error">{error}</p></div></main>;
  if (!user) return <main className="auth-page"><div className="auth-loading"><span className="spinner"/>FiberVision 불러오는 중</div></main>;
  return <>{children(user)}</>;
}
