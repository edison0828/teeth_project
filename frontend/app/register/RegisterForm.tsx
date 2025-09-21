"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

import { loginUser, registerUser } from "../../lib/api";
import { persistToken } from "../../lib/auth-storage";
import { useAuth } from "../../contexts/AuthContext";

export default function RegisterForm() {
  const router = useRouter();
  const { refresh, token: activeToken, loading: authLoading } = useAuth();
  const [email, setEmail] = useState<string>("");
  const [fullName, setFullName] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [confirmPassword, setConfirmPassword] = useState<string>("");
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

    if (password !== confirmPassword) {
      setStatus("兩次輸入的密碼不相同");
      return;
    }

    setStatus("");
    setLoading(true);

    try {
      await registerUser({ email: email.trim(), password, full_name: fullName || undefined });
      const token = await loginUser(email.trim(), password);
      persistToken(token.access_token, token.expires_in);
      await refresh();
      setStatus("註冊成功，帳號已自動登入");
      router.replace("/");
    } catch (error) {
      const message = error instanceof Error ? error.message : "註冊失敗";
      setStatus(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto mt-16 max-w-md rounded-3xl bg-white/5 p-8 shadow-card">
      <h1 className="text-2xl font-semibold text-white">建立新帳號</h1>
      <p className="mt-2 text-sm text-slate-400">填寫以下資訊即可註冊並立即開始使用平台。</p>
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
          <label className="text-xs uppercase tracking-wide text-slate-400">姓名</label>
          <input
            type="text"
            autoComplete="name"
            value={fullName}
            onChange={(event) => setFullName(event.target.value)}
            className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
            placeholder="可略過"
          />
        </div>
        <div>
          <label className="text-xs uppercase tracking-wide text-slate-400">密碼</label>
          <input
            type="password"
            autoComplete="new-password"
            required
            minLength={8}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
          />
        </div>
        <div>
          <label className="text-xs uppercase tracking-wide text-slate-400">再次輸入密碼</label>
          <input
            type="password"
            required
            value={confirmPassword}
            onChange={(event) => setConfirmPassword(event.target.value)}
            className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
          />
        </div>
        {status && <p className={`text-xs ${status.includes("成功") ? "text-accent" : "text-red-400"}`}>{status}</p>}
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-full bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40 disabled:opacity-60"
        >
          {loading ? "送出中..." : "註冊"}
        </button>
      </form>
      <p className="mt-6 text-center text-xs text-slate-400">
        已經有帳號了？
        <Link className="ml-1 text-primary" href="/login">
          立刻登入
        </Link>
      </p>
    </div>
  );
}