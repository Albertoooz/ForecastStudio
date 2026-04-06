"use client";

import { useState } from "react";
import axios from "axios";
import { authApi } from "@/lib/api";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"login" | "register">("login");
  const [fullName, setFullName] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (mode === "login") {
        const res = await authApi.login(email, password);
        localStorage.setItem("access_token", res.data.access_token);
      } else {
        await authApi.register(email, password, fullName);
        const res = await authApi.login(email, password);
        localStorage.setItem("access_token", res.data.access_token);
      }
      window.location.href = "/dashboard";
    } catch (err: unknown) {
      if (!axios.isAxiosError(err)) {
        setError("Something went wrong. Please try again.");
        return;
      }
      const data = err.response?.data as { detail?: string | string[] } | undefined;
      const detail = data?.detail;
      const message = Array.isArray(detail)
        ? detail.join(", ")
        : typeof detail === "string"
          ? detail
          : err.message || (mode === "login"
              ? "Invalid email or password."
              : "Registration failed. Email may already be in use.");
      const noResponse = err.response === undefined;
      setError(
        noResponse || err.response?.status === 0 || err.message === "Network Error"
          ? "Cannot reach server. Is the backend running at " + (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "?"
          : message
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 dark:bg-gray-950 px-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="mb-8 flex flex-col items-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-brand-600 text-white font-bold text-xl mb-4">
            F
          </div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Forecaster</h1>
          <p className="mt-1 text-sm text-gray-500">
            {mode === "login" ? "Sign in to your account" : "Create a new account"}
          </p>
        </div>

        {/* Card */}
        <div className="rounded-2xl border bg-white p-8 shadow-sm dark:bg-gray-900 dark:border-gray-800">
          <form onSubmit={handleSubmit} className="space-y-5">
            {mode === "register" && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                  Full name
                </label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  required
                  placeholder="Your name"
                  className="w-full rounded-xl border bg-gray-50 px-4 py-2.5 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 dark:bg-gray-800 dark:border-gray-700 dark:text-white"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="you@example.com"
                className="w-full rounded-xl border bg-gray-50 px-4 py-2.5 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 dark:bg-gray-800 dark:border-gray-700 dark:text-white"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="••••••••"
                className="w-full rounded-xl border bg-gray-50 px-4 py-2.5 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 dark:bg-gray-800 dark:border-gray-700 dark:text-white"
              />
            </div>

            {error && (
              <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700 dark:bg-red-900/20 dark:border-red-800 dark:text-red-400">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-brand-600 py-2.5 text-sm font-medium text-white transition-colors hover:bg-brand-700 disabled:opacity-50"
            >
              {loading
                ? "Please wait..."
                : mode === "login"
                ? "Sign in"
                : "Create account"}
            </button>
          </form>

          <div className="mt-6 text-center text-sm text-gray-500">
            {mode === "login" ? (
              <>
                Don&apos;t have an account?{" "}
                <button
                  onClick={() => { setMode("register"); setError(""); }}
                  className="font-medium text-brand-600 hover:underline"
                >
                  Sign up
                </button>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <button
                  onClick={() => { setMode("login"); setError(""); }}
                  className="font-medium text-brand-600 hover:underline"
                >
                  Sign in
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
