"use client";

import { useEffect, useState, useCallback } from "react";
import { AppShell } from "@/components/layout/AppShell";
import {
  Activity, AlertTriangle, TrendingDown, Shield,
  Calendar, Loader2, ToggleLeft, ToggleRight, Trash2,
  Plus, RefreshCw,
} from "lucide-react";
import { monitoringApi } from "@/lib/api";
import type { MonitoringSummary, Schedule } from "@/types";

const SCHEDULE_TYPES = [
  { value: "retrain", label: "Retrain" },
  { value: "forecast", label: "Forecast" },
  { value: "monitor", label: "Monitor" },
];

const CRON_PRESETS = [
  { label: "Daily at 2am", value: "0 2 * * *" },
  { label: "Weekly (Mon 2am)", value: "0 2 * * 1" },
  { label: "Monthly (1st 2am)", value: "0 2 1 * *" },
];

export default function MonitoringPage() {
  const [summary, setSummary] = useState<MonitoringSummary | null>(null);
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const [showNewSchedule, setShowNewSchedule] = useState(false);
  const [scheduleForm, setScheduleForm] = useState({
    model_version_id: "",
    schedule_type: "retrain",
    cron_expression: "0 2 * * 1",
  });
  const [formError, setFormError] = useState<string | null>(null);
  const [formSaving, setFormSaving] = useState(false);

  const load = useCallback(async (silent = false) => {
    if (!silent) setRefreshing(true);
    try {
      const [sumRes, schedRes] = await Promise.allSettled([
        monitoringApi.summary(),
        monitoringApi.listSchedules(),
      ]);
      if (sumRes.status === "fulfilled") setSummary(sumRes.value.data);
      if (schedRes.status === "fulfilled") setSchedules(schedRes.value.data ?? []);
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleToggle = async (schedule: Schedule) => {
    try {
      const res = await monitoringApi.toggleSchedule(schedule.id);
      setSchedules((prev) =>
        prev.map((s) => (s.id === schedule.id ? res.data : s))
      );
    } catch {
      /* noop */
    }
  };

  const handleDelete = async (schedule: Schedule) => {
    if (!confirm("Delete this schedule?")) return;
    try {
      await monitoringApi.deleteSchedule(schedule.id);
      setSchedules((prev) => prev.filter((s) => s.id !== schedule.id));
    } catch {
      /* noop */
    }
  };

  const handleCreateSchedule = async () => {
    if (!scheduleForm.model_version_id) {
      setFormError("Model version ID is required.");
      return;
    }
    setFormSaving(true);
    setFormError(null);
    try {
      const res = await monitoringApi.createSchedule(scheduleForm);
      setSchedules((prev) => [res.data, ...prev]);
      setShowNewSchedule(false);
    } catch (err: unknown) {
      setFormError(
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
          "Failed to create schedule."
      );
    } finally {
      setFormSaving(false);
    }
  };

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Monitoring</h1>
            <p className="text-sm text-gray-500">
              Model drift, alerts and automated schedules
            </p>
          </div>
          <button
            onClick={() => load()}
            disabled={refreshing}
            className="flex items-center gap-2 rounded-xl border px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:border-gray-700 dark:hover:bg-gray-800 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>

        {/* Summary cards */}
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <SummaryCard
            icon={Shield}
            iconCls="bg-green-50 text-green-600 dark:bg-green-900/20"
            label="Active Models"
            value={loading ? null : summary?.active_versions ?? 0}
          />
          <SummaryCard
            icon={AlertTriangle}
            iconCls="bg-red-50 text-red-600 dark:bg-red-900/20"
            label="Alerts (24h)"
            value={loading ? null : summary?.total_alerts_24h ?? 0}
          />
          <SummaryCard
            icon={TrendingDown}
            iconCls="bg-yellow-50 text-yellow-600 dark:bg-yellow-900/20"
            label="Total Versions"
            value={loading ? null : summary?.total_model_versions ?? 0}
          />
          <SummaryCard
            icon={Activity}
            iconCls="bg-brand-50 text-brand-600 dark:bg-brand-900/20"
            label="Active Schedules"
            value={loading ? null : summary?.active_schedules ?? 0}
          />
        </div>

        {/* Recent metrics */}
        {summary && summary.recent_metrics.length > 0 && (
          <div className="rounded-xl border bg-white p-6 shadow-sm dark:bg-gray-900 dark:border-gray-800">
            <h2 className="mb-4 font-semibold">Recent Monitoring Metrics</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="border-b dark:border-gray-700">
                  <tr className="text-left text-xs font-medium text-gray-500 uppercase tracking-wide">
                    <th className="pb-2 pr-6">Metric</th>
                    <th className="pb-2 pr-6">Value</th>
                    <th className="pb-2 pr-6">Threshold</th>
                    <th className="pb-2 pr-6">Alert</th>
                    <th className="pb-2">Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y dark:divide-gray-800">
                  {summary.recent_metrics.map((m, i) => (
                    <tr key={i} className={m.alert_triggered ? "bg-red-50 dark:bg-red-900/10" : ""}>
                      <td className="py-2 pr-6 font-medium">{m.metric_name}</td>
                      <td className="py-2 pr-6 font-mono">{m.metric_value.toFixed(4)}</td>
                      <td className="py-2 pr-6 font-mono text-gray-400">
                        {m.threshold != null ? m.threshold.toFixed(4) : "—"}
                      </td>
                      <td className="py-2 pr-6">
                        {m.alert_triggered ? (
                          <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700">
                            <AlertTriangle className="h-3 w-3" /> Alert
                          </span>
                        ) : (
                          <span className="text-green-600 text-xs">OK</span>
                        )}
                      </td>
                      <td className="py-2 text-xs text-gray-400">
                        {new Date(m.timestamp).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Schedules */}
        <div className="rounded-xl border bg-white shadow-sm dark:bg-gray-900 dark:border-gray-800">
          <div className="flex items-center justify-between border-b px-6 py-4 dark:border-gray-800">
            <h2 className="font-semibold">Schedules ({schedules.length})</h2>
            <button
              onClick={() => setShowNewSchedule((v) => !v)}
              className="flex items-center gap-1 rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-brand-700"
            >
              <Plus className="h-4 w-4" />
              New Schedule
            </button>
          </div>

          {/* New schedule form */}
          {showNewSchedule && (
            <div className="border-b bg-gray-50 px-6 py-5 dark:border-gray-800 dark:bg-gray-800/40">
              <h3 className="mb-4 text-sm font-semibold">New Schedule</h3>
              <div className="grid gap-4 sm:grid-cols-3">
                <label className="space-y-1">
                  <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                    Model Version ID
                  </span>
                  <input
                    type="text"
                    placeholder="UUID of promoted version"
                    value={scheduleForm.model_version_id}
                    onChange={(e) =>
                      setScheduleForm((s) => ({ ...s, model_version_id: e.target.value }))
                    }
                    className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
                  />
                </label>

                <label className="space-y-1">
                  <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                    Type
                  </span>
                  <select
                    value={scheduleForm.schedule_type}
                    onChange={(e) =>
                      setScheduleForm((s) => ({ ...s, schedule_type: e.target.value }))
                    }
                    className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
                  >
                    {SCHEDULE_TYPES.map((t) => (
                      <option key={t.value} value={t.value}>{t.label}</option>
                    ))}
                  </select>
                </label>

                <label className="space-y-1">
                  <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                    Cron expression
                  </span>
                  <input
                    type="text"
                    value={scheduleForm.cron_expression}
                    onChange={(e) =>
                      setScheduleForm((s) => ({ ...s, cron_expression: e.target.value }))
                    }
                    className="w-full rounded-lg border px-3 py-2 font-mono text-sm dark:bg-gray-900 dark:border-gray-700"
                  />
                  <div className="flex gap-2">
                    {CRON_PRESETS.map((p) => (
                      <button
                        key={p.value}
                        onClick={() =>
                          setScheduleForm((s) => ({ ...s, cron_expression: p.value }))
                        }
                        className="rounded bg-gray-200 px-1.5 py-0.5 text-xs hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600"
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </label>
              </div>

              {formError && (
                <p className="mt-2 text-sm text-red-600">{formError}</p>
              )}

              <div className="mt-4 flex gap-2">
                <button
                  onClick={handleCreateSchedule}
                  disabled={formSaving}
                  className="flex items-center gap-1 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
                >
                  {formSaving && <Loader2 className="h-3 w-3 animate-spin" />}
                  Create
                </button>
                <button
                  onClick={() => { setShowNewSchedule(false); setFormError(null); }}
                  className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-5 w-5 animate-spin text-gray-300" />
            </div>
          ) : schedules.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-gray-400">
              <Calendar className="mb-3 h-10 w-10" />
              <p className="text-sm">No schedules — create one above</p>
            </div>
          ) : (
            <div className="divide-y dark:divide-gray-800">
              {schedules.map((s) => (
                <div
                  key={s.id}
                  className="flex items-center justify-between px-6 py-4 hover:bg-gray-50 dark:hover:bg-gray-800/50"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span
                        className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                          s.is_active
                            ? "bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400"
                            : "bg-gray-50 text-gray-500 dark:bg-gray-800"
                        }`}
                      >
                        {s.schedule_type}
                      </span>
                      <code className="text-xs text-gray-500">{s.cron_expression}</code>
                    </div>
                    <p className="mt-0.5 text-xs text-gray-400">
                      Version {s.model_version_id.slice(0, 8)}…
                      {s.next_run_at && ` • next: ${new Date(s.next_run_at).toLocaleString()}`}
                      {s.last_run_at && ` • last: ${new Date(s.last_run_at).toLocaleString()}`}
                    </p>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleToggle(s)}
                      title={s.is_active ? "Deactivate" : "Activate"}
                      className="text-gray-400 hover:text-brand-600"
                    >
                      {s.is_active ? (
                        <ToggleRight className="h-6 w-6 text-brand-600" />
                      ) : (
                        <ToggleLeft className="h-6 w-6" />
                      )}
                    </button>
                    <button
                      onClick={() => handleDelete(s)}
                      className="rounded-lg p-2 text-gray-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-900/20"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}

function SummaryCard({
  icon: Icon,
  iconCls,
  label,
  value,
}: {
  icon: React.ElementType;
  iconCls: string;
  label: string;
  value: number | null;
}) {
  return (
    <div className="rounded-xl border bg-white p-6 shadow-sm dark:bg-gray-900 dark:border-gray-800">
      <div className="flex items-center gap-3">
        <div className={`rounded-lg p-2.5 ${iconCls}`}>
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className="text-xl font-bold">
            {value === null ? (
              <Loader2 className="h-5 w-5 animate-spin text-gray-300" />
            ) : (
              value
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
