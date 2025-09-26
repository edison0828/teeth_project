"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

import { loginUser } from "../../lib/api";
import { persistToken } from "../../lib/auth-storage";
import { useAuth } from "../../contexts/AuthContext";

export default function LoginForm() {
  const router = useRouter();
  const {
    refresh,
    token: activeToken,
    loading: authLoading,
    guestMode,
    enterGuestMode,
    exitGuestMode,
  } = useAuth();
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (!authLoading && activeToken) {
      router.replace("/");
    }
  }, [authLoading, activeToken, router]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (loading) {
      return;
    }

    setStatus("");
    setLoading(true);

    try {
      const token = await loginUser(email.trim(), password);
      persistToken(token.access_token, token.expires_in);
      exitGuestMode();
      await refresh();
      setStatus("登入成功，稍後自動導向");
      router.replace("/");
    } catch (error) {
      const message = error instanceof Error ? error.message : "登入失敗";
      setStatus(message);
    } finally {
      setLoading(false);
    }
  };

  const handleGuestMode = () => {
    if (loading) {
      return;
    }
    enterGuestMode();
    setStatus("已啟用示範模式，即將導向體驗儀表板。");
    router.replace("/");
  };

  return (
    <div className="mx-auto mt-16 max-w-md rounded-3xl bg-white/5 p-8 shadow-card">
      <h1 className="text-2xl font-semibold text-white">登入口腔 X 光平台</h1>
      <p className="mt-2 text-sm text-slate-400">輸入註冊時的電子郵件與密碼即可登入。</p>
      <form onSubmit={handleSubmit} className="mt-6 space-y-5 text-sm text-slate-200">
        <div>
          <label className="text-xs uppercase tracking-wide text-slate-400">電子郵件</label>
          <input
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
          />
        </div>
        <div>
          <label className="text-xs uppercase tracking-wide text-slate-400">密碼</label>
          <input
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
          />
        </div>
        {status && (
          <p
            className={`text-xs ${
              status.includes("成功") || status.includes("啟用")
                ? "text-accent"
                : "text-red-400"
            }`}
          >
            {status}
          </p>
        )}
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-full bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40 disabled:opacity-60"
        >
          {loading ? "登入中..." : "登入"}
        </button>
      </form>
      <button
        type="button"
        onClick={handleGuestMode}
        className="mt-4 w-full rounded-full border border-white/10 bg-transparent px-4 py-3 text-sm font-semibold text-slate-100 transition hover:border-primary hover:text-primary"
      >
        {guestMode ? "繼續體驗示範模式" : "不登入，直接試用示範模式"}
      </button>
      <p className="mt-6 text-center text-xs text-slate-400">
        還沒有帳號嗎？
        <Link className="ml-1 text-primary" href="/register">
          註冊一個
        </Link>
      </p>
    </div>
  );
}