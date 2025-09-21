'use client';

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

import {
  changePassword,
  fetchCurrentUser,
  logoutUser,
  updateProfile
} from "../../lib/api";
import { clearToken, readToken } from "../../lib/auth-storage";
import type { UserProfile } from "../../lib/types";

export default function AccountPanel() {
  const router = useRouter();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [message, setMessage] = useState<string>("");
  const [passwordMessage, setPasswordMessage] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [updating, setUpdating] = useState<boolean>(false);
  const [changing, setChanging] = useState<boolean>(false);
  const [fullName, setFullName] = useState<string>("");
  const [email, setEmail] = useState<string>("");
  const [currentPassword, setCurrentPassword] = useState<string>("");
  const [newPassword, setNewPassword] = useState<string>("");
  const [confirmPassword, setConfirmPassword] = useState<string>("");

  useEffect(() => {
    const token = readToken();
    if (!token) {
      router.replace("/login");
      return;
    }

    fetchCurrentUser(token)
      .then((user) => {
        setProfile(user);
        setFullName(user.full_name ?? "");
        setEmail(user.email);
      })
      .catch((error) => {
        clearToken();
        const message = error instanceof Error ? error.message : "讀取帳號資料失敗";
        setMessage(message);
        router.replace("/login");
      })
      .finally(() => {
        setLoading(false);
      });
  }, [router]);

  const handleProfileSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const token = readToken();
    if (!token || !profile) {
      router.replace("/login");
      return;
    }

    setMessage("");
    setUpdating(true);
    try {
      const updated = await updateProfile(token, { email: email.trim(), full_name: fullName.trim() || undefined });
      setProfile(updated);
      setMessage("資料已更新");
    } catch (error) {
      const msg = error instanceof Error ? error.message : "更新失敗";
      setMessage(msg);
    } finally {
      setUpdating(false);
    }
  };

  const handlePasswordSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const token = readToken();
    if (!token) {
      router.replace("/login");
      return;
    }

    if (newPassword !== confirmPassword) {
      setPasswordMessage("新密碼與確認密碼不一致");
      return;
    }

    setPasswordMessage("");
    setChanging(true);
    try {
      await changePassword(token, { current_password: currentPassword, new_password: newPassword });
      setPasswordMessage("密碼更新完成");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (error) {
      const msg = error instanceof Error ? error.message : "變更密碼失敗";
      setPasswordMessage(msg);
    } finally {
      setChanging(false);
    }
  };

  const handleLogout = async () => {
    const token = readToken();
    if (token) {
      try {
        await logoutUser(token);
      } catch (error) {
        console.warn("Logout warning", error);
      }
    }
    clearToken();
    router.replace("/login");
  };

  if (loading) {
    return <p className="mt-16 text-center text-sm text-slate-300">載入帳號資料中...</p>;
  }

  if (!profile) {
    return null;
  }

  return (
    <div className="mx-auto mt-10 grid max-w-4xl gap-8 md:grid-cols-2">
      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <h2 className="text-xl font-semibold text-white">基本資料</h2>
        <p className="mt-1 text-sm text-slate-400">更新您的顯示名稱或通知電子郵件。</p>
        <form onSubmit={handleProfileSubmit} className="mt-6 space-y-4 text-sm text-slate-200">
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">電子郵件</label>
            <input
              type="email"
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
              value={fullName}
              onChange={(event) => setFullName(event.target.value)}
              className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
              placeholder="選填"
            />
          </div>
          {message && <p className={`text-xs ${message.includes("更新") ? "text-accent" : "text-red-400"}`}>{message}</p>}
          <button
            type="submit"
            disabled={updating}
            className="w-full rounded-full bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40 disabled:opacity-60"
          >
            {updating ? "儲存中..." : "儲存設定"}
          </button>
        </form>
      </section>

      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <h2 className="text-xl font-semibold text-white">變更密碼</h2>
        <p className="mt-1 text-sm text-slate-400">建議定期更新密碼以提升安全性。</p>
        <form onSubmit={handlePasswordSubmit} className="mt-6 space-y-4 text-sm text-slate-200">
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">目前密碼</label>
            <input
              type="password"
              required
              value={currentPassword}
              onChange={(event) => setCurrentPassword(event.target.value)}
              className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
            />
          </div>
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">新密碼</label>
            <input
              type="password"
              required
              minLength={8}
              value={newPassword}
              onChange={(event) => setNewPassword(event.target.value)}
              className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
            />
          </div>
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">確認新密碼</label>
            <input
              type="password"
              required
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
            />
          </div>
          {passwordMessage && (
            <p className={`text-xs ${passwordMessage.includes("完成") ? "text-accent" : "text-red-400"}`}>
              {passwordMessage}
            </p>
          )}
          <button
            type="submit"
            disabled={changing}
            className="w-full rounded-full bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40 disabled:opacity-60"
          >
            {changing ? "送出中..." : "更新密碼"}
          </button>
        </form>
        <button
          onClick={handleLogout}
          className="mt-6 w-full rounded-full border border-red-400/40 px-4 py-3 text-sm font-semibold text-red-300 hover:bg-red-500/10"
        >
          登出所有裝置
        </button>
      </section>
    </div>
  );
}
