"use client";

import { useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { Clock, Play, Pause, Trash2, Plus } from "lucide-react";
import type { Schedule } from "@/types";

export default function SchedulingPage() {
  const [schedules, setSchedules] = useState<Schedule[]>([]);

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Scheduling</h1>
            <p className="text-sm text-gray-500">
              Automate retraining, forecasting and monitoring on a schedule
            </p>
          </div>
          <button className="flex items-center gap-2 rounded-xl bg-brand-600 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-brand-700">
            <Plus className="h-4 w-4" />
            New schedule
          </button>
        </div>

        {/* Schedule list */}
        <div className="rounded-xl border bg-white shadow-sm dark:bg-gray-900 dark:border-gray-800">
          <div className="border-b px-6 py-4 dark:border-gray-800">
            <h2 className="font-semibold">Active schedules</h2>
          </div>

          {schedules.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-gray-400">
              <Clock className="mb-4 h-12 w-12" />
              <p className="text-sm">No schedules yet — click &quot;New schedule&quot;</p>
            </div>
          ) : (
            <div className="divide-y dark:divide-gray-800">
              {schedules.map((s) => (
                <div
                  key={s.id}
                  className="flex items-center justify-between px-6 py-4"
                >
                  <div>
                    <p className="font-medium">{s.schedule_type}</p>
                    <p className="text-xs text-gray-500">
                      Cron: {s.cron_expression}
                      {s.next_run_at && ` • Next: ${s.next_run_at}`}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                        s.is_active
                          ? "bg-green-50 text-green-700"
                          : "bg-gray-100 text-gray-500"
                      }`}
                    >
                      {s.is_active ? "Active" : "Paused"}
                    </span>
                    <button className="rounded p-1 hover:bg-gray-100 dark:hover:bg-gray-800">
                      {s.is_active ? (
                        <Pause className="h-4 w-4 text-gray-500" />
                      ) : (
                        <Play className="h-4 w-4 text-gray-500" />
                      )}
                    </button>
                    <button className="rounded p-1 hover:bg-gray-100 dark:hover:bg-gray-800">
                      <Trash2 className="h-4 w-4 text-red-500" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Help */}
        <div className="rounded-xl border border-brand-200 bg-brand-50 p-6 dark:bg-brand-900/10 dark:border-brand-800">
          <h3 className="font-semibold text-brand-900 dark:text-brand-300">
            Schedule types
          </h3>
          <ul className="mt-2 space-y-1 text-sm text-brand-700 dark:text-brand-400">
            <li>
              <strong>Retrain</strong> — automatic model retraining on new data
            </li>
            <li>
              <strong>Forecast</strong> — periodic forecast generation
            </li>
            <li>
              <strong>Monitor</strong> — drift and data quality checks
            </li>
          </ul>
        </div>
      </div>
    </AppShell>
  );
}
