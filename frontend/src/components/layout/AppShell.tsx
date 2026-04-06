"use client";

import { ReactNode } from "react";
import { Sidebar } from "./Sidebar";

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-950">
        <div className="mx-auto max-w-7xl p-6">{children}</div>
      </main>
    </div>
  );
}
