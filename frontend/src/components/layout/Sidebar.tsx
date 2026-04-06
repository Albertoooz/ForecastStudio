"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  MessageSquare,
  Database,
  BarChart3,
  GitBranch,
  Activity,
  Clock,
  LogOut,
} from "lucide-react";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/data", label: "Training Data", icon: Database },
  { href: "/analysis", label: "Model Analysis", icon: BarChart3 },
  { href: "/pipeline", label: "Pipeline", icon: GitBranch },
  { href: "/monitoring", label: "Monitoring", icon: Activity },
  { href: "/scheduling", label: "Scheduling", icon: Clock },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-64 flex-col border-r bg-[var(--sidebar-bg)] border-[var(--sidebar-border)]">
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 border-b border-[var(--sidebar-border)] px-6">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-white font-bold text-sm">
          F
        </div>
        <span className="text-lg font-semibold">Forecaster</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-4">
        <ul className="space-y-1">
          {navItems.map(({ href, label, icon: Icon }) => {
            const isActive = pathname.startsWith(href);
            return (
              <li key={href}>
                <Link
                  href={href}
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-brand-50 text-brand-700 dark:bg-brand-900/30 dark:text-brand-300"
                      : "text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-200"
                  )}
                >
                  <Icon className="h-5 w-5" />
                  {label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Footer */}
      <div className="border-t border-[var(--sidebar-border)] p-4">
        <button
          onClick={() => {
            localStorage.removeItem("access_token");
            window.location.href = "/login";
          }}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800"
        >
          <LogOut className="h-5 w-5" />
          Sign out
        </button>
      </div>
    </aside>
  );
}
