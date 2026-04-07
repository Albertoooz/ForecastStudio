"use client";

import { ExternalLink } from "lucide-react";

import { AppShell } from "@/components/layout/AppShell";

const DAGSTER_URL =
  process.env.NEXT_PUBLIC_DAGSTER_URL || "http://localhost:3005";

/**
 * Scheduling = embedded Dagster UI. All schedules, sensors, and ops are
 * managed in Dagster; this page is a thin shell so users stay inside Forecast Studio.
 */
export default function SchedulingPage() {
  return (
    <AppShell>
      <div className="-mx-6 -mt-2 flex min-h-[calc(100vh-5rem)] flex-col">
        <div className="mb-4 px-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold">Scheduling</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Pipelines and schedules live in{" "}
                <span className="font-medium text-gray-700 dark:text-gray-300">
                  Dagster
                </span>
                — edit runs, sensors, and jobs here without leaving the app.
              </p>
            </div>
            <a
              href={DAGSTER_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm transition-colors hover:bg-gray-50 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200 dark:hover:bg-gray-800"
            >
              <ExternalLink className="h-4 w-4" />
              Open Dagster in new tab
            </a>
          </div>
        </div>

        <div className="min-h-0 flex-1 px-6 pb-6">
          <div className="h-[min(78vh,calc(100vh-11rem))] overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900">
            <iframe
              title="Dagster"
              src={DAGSTER_URL}
              className="h-full w-full border-0"
              referrerPolicy="no-referrer-when-downgrade"
            />
          </div>
          <p className="mt-3 text-xs text-gray-500 dark:text-gray-400">
            If the frame is empty, Dagster may block embedding (
            <code className="rounded bg-gray-100 px-1 dark:bg-gray-800">
              X-Frame-Options
            </code>
            ). Use &quot;Open Dagster in new tab&quot; or set{" "}
            <code className="rounded bg-gray-100 px-1 dark:bg-gray-800">
              NEXT_PUBLIC_DAGSTER_URL
            </code>{" "}
            to match your Dagster host.
          </p>
        </div>
      </div>
    </AppShell>
  );
}
