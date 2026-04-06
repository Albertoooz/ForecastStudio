"use client";

import { useEffect, useState, useCallback } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { BarChart3, TrendingUp, Target, Gauge, Loader2, ChevronDown } from "lucide-react";
import { modelsApi } from "@/lib/api";
import type { ModelRun } from "@/types";

function MetricBar({
  label,
  value,
  baseline,
  unit = "",
  lowerIsBetter = true,
}: {
  label: string;
  value: number;
  baseline?: number;
  unit?: string;
  lowerIsBetter?: boolean;
}) {
  const improvement =
    baseline != null && baseline > 0
      ? ((baseline - value) / baseline) * 100
      : null;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-gray-700 dark:text-gray-300">{label}</span>
        <span className="font-mono text-gray-900 dark:text-white">
          {value.toFixed(4)}{unit}
        </span>
      </div>
      {baseline != null && improvement != null && (
        <p className={`text-xs ${improvement > 0 ? "text-green-600" : "text-red-500"}`}>
          {improvement > 0 ? "▼" : "▲"} {Math.abs(improvement).toFixed(1)}%
          {lowerIsBetter
            ? improvement > 0 ? " better than naive" : " worse than naive"
            : improvement > 0 ? " improvement" : " decline"}
        </p>
      )}
    </div>
  );
}

function HealthBadge({ score }: { score: number }) {
  const color =
    score >= 70 ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
    : score >= 40 ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400"
    : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400";

  return (
    <span className={`rounded-full px-3 py-1 text-sm font-semibold ${color}`}>
      Health: {score}/100
    </span>
  );
}

export default function AnalysisPage() {
  const [runs, setRuns] = useState<ModelRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<ModelRun | null>(null);
  const [forecast, setForecast] = useState<Record<string, unknown> | null>(null);
  const [fcLoading, setFcLoading] = useState(false);

  const load = useCallback(async () => {
    try {
      const res = await modelsApi.listRuns({ limit: 50, status: "completed" });
      const completed: ModelRun[] = (res.data.runs ?? []).filter(
        (r: ModelRun) => r.status === "completed"
      );
      setRuns(completed);
      if (completed.length > 0 && !selected) {
        setSelected(completed[0]);
      }
    } finally {
      setLoading(false);
    }
  }, [selected]);

  useEffect(() => { load(); }, [load]);

  const loadForecast = useCallback(async (run: ModelRun) => {
    setFcLoading(true);
    setForecast(null);
    try {
      const res = await modelsApi.predict(run.id);
      setForecast(res.data.predictions);
    } catch {
      /* noop */
    } finally {
      setFcLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selected) loadForecast(selected);
  }, [selected, loadForecast]);

  const m = selected?.metrics;

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Model Analysis</h1>
            <p className="text-sm text-gray-500">
              Metrics, forecast preview and model comparison
            </p>
          </div>

          {/* Run selector */}
          {runs.length > 0 && (
            <div className="relative">
              <select
                value={selected?.id ?? ""}
                onChange={(e) => {
                  const r = runs.find((r) => r.id === e.target.value);
                  if (r) setSelected(r);
                }}
                className="appearance-none rounded-xl border bg-white pl-4 pr-8 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
              >
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.best_model_name || r.model_type} — {new Date(r.created_at).toLocaleDateString()}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            </div>
          )}
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-24">
            <Loader2 className="h-8 w-8 animate-spin text-gray-300" />
          </div>
        ) : runs.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border bg-white py-24 text-gray-400 dark:bg-gray-900 dark:border-gray-800">
            <BarChart3 className="mb-4 h-12 w-12" />
            <p className="text-sm">No completed runs yet — train a model first</p>
          </div>
        ) : selected ? (
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Metrics */}
            <div className="rounded-xl border bg-white p-6 shadow-sm dark:bg-gray-900 dark:border-gray-800">
              <div className="mb-5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-brand-600" />
                  <h2 className="font-semibold">Model Metrics</h2>
                </div>
                {m?.health_score != null && (
                  <HealthBadge score={m.health_score} />
                )}
              </div>

              {m ? (
                <div className="space-y-4">
                  {m.rmse != null && (
                    <MetricBar
                      label="RMSE"
                      value={m.rmse}
                      baseline={m.rmse_naive}
                    />
                  )}
                  {m.mape != null && (
                    <MetricBar label="MAPE" value={m.mape} unit="%" />
                  )}
                  {m.rmse_improvement_pct != null && (
                    <div className="rounded-lg bg-green-50 p-3 dark:bg-green-900/20">
                      <p className="text-sm font-medium text-green-700 dark:text-green-400">
                        {m.rmse_improvement_pct.toFixed(1)}% RMSE improvement vs naive baseline
                      </p>
                    </div>
                  )}
                  {m.n_predictions != null && (
                    <p className="text-xs text-gray-400">{m.n_predictions} predictions generated</p>
                  )}
                </div>
              ) : (
                <p className="text-sm text-gray-500">No metrics recorded for this run</p>
              )}

              <div className="mt-4 border-t pt-4 dark:border-gray-800">
                <p className="text-xs text-gray-400">
                  Best model: <span className="font-medium text-gray-600 dark:text-gray-300">{selected.best_model_name || selected.model_type}</span>
                  {" "}• Horizon: {selected.horizon}
                  {selected.duration_seconds != null && ` • Trained in ${selected.duration_seconds.toFixed(0)}s`}
                </p>
              </div>
            </div>

            {/* Forecast preview */}
            <div className="rounded-xl border bg-white p-6 shadow-sm dark:bg-gray-900 dark:border-gray-800">
              <div className="mb-4 flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-brand-600" />
                <h2 className="font-semibold">Forecast Preview</h2>
              </div>

              {fcLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-5 w-5 animate-spin text-gray-300" />
                </div>
              ) : forecast && typeof forecast === "object" && !Array.isArray(forecast) ? (
                <ForecastTable data={forecast as { dates?: string[]; predictions?: number[] }} />
              ) : (
                <p className="text-sm text-gray-500">No forecast data available yet</p>
              )}
            </div>

            {/* Model comparison (all completed runs) */}
            <div className="rounded-xl border bg-white p-6 shadow-sm dark:bg-gray-900 dark:border-gray-800 lg:col-span-2">
              <div className="mb-4 flex items-center gap-2">
                <Gauge className="h-5 w-5 text-brand-600" />
                <h2 className="font-semibold">All Completed Runs</h2>
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="border-b dark:border-gray-700">
                    <tr className="text-left text-xs font-medium text-gray-500 uppercase tracking-wide">
                      <th className="pb-2 pr-6">Model</th>
                      <th className="pb-2 pr-6">RMSE</th>
                      <th className="pb-2 pr-6">MAPE</th>
                      <th className="pb-2 pr-6">Improvement</th>
                      <th className="pb-2 pr-6">Health</th>
                      <th className="pb-2">Date</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y dark:divide-gray-800">
                    {runs.map((r) => (
                      <tr
                        key={r.id}
                        onClick={() => setSelected(r)}
                        className={`cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 ${
                          r.id === selected?.id ? "bg-brand-50 dark:bg-brand-900/10" : ""
                        }`}
                      >
                        <td className="py-2 pr-6 font-medium">
                          {r.best_model_name || r.model_type}
                        </td>
                        <td className="py-2 pr-6 font-mono">
                          {r.metrics?.rmse?.toFixed(4) ?? "—"}
                        </td>
                        <td className="py-2 pr-6 font-mono">
                          {r.metrics?.mape != null ? `${r.metrics.mape.toFixed(2)}%` : "—"}
                        </td>
                        <td className="py-2 pr-6">
                          {r.metrics?.rmse_improvement_pct != null ? (
                            <span
                              className={
                                r.metrics.rmse_improvement_pct > 0
                                  ? "text-green-600"
                                  : "text-red-500"
                              }
                            >
                              {r.metrics.rmse_improvement_pct > 0 ? "+" : ""}
                              {r.metrics.rmse_improvement_pct.toFixed(1)}%
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                        <td className="py-2 pr-6">
                          {r.metrics?.health_score != null ? (
                            <span
                              className={
                                r.metrics.health_score >= 70
                                  ? "text-green-600"
                                  : r.metrics.health_score >= 40
                                  ? "text-yellow-600"
                                  : "text-red-500"
                              }
                            >
                              {r.metrics.health_score}/100
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                        <td className="py-2 text-xs text-gray-400">
                          {new Date(r.created_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </AppShell>
  );
}

function ForecastTable({
  data,
}: {
  data: { dates?: string[]; predictions?: number[] };
}) {
  if (!data.dates || !data.predictions || data.dates.length === 0) {
    return <p className="text-sm text-gray-500">No predictions found</p>;
  }

  return (
    <div className="overflow-y-auto max-h-64">
      <table className="min-w-full text-sm">
        <thead className="sticky top-0 bg-white dark:bg-gray-900">
          <tr className="border-b dark:border-gray-700">
            <th className="pb-2 pr-6 text-left text-xs font-medium text-gray-500 uppercase">#</th>
            <th className="pb-2 pr-6 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
            <th className="pb-2 text-left text-xs font-medium text-gray-500 uppercase">Prediction</th>
          </tr>
        </thead>
        <tbody className="divide-y dark:divide-gray-800">
          {data.dates.slice(0, 50).map((d, i) => (
            <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
              <td className="py-1.5 pr-6 text-gray-400 text-xs">{i + 1}</td>
              <td className="py-1.5 pr-6 font-mono text-xs">{d}</td>
              <td className="py-1.5 font-mono font-semibold">
                {data.predictions![i].toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.dates.length > 50 && (
        <p className="px-2 py-1 text-xs text-gray-400">
          Showing 50 of {data.dates.length} predictions
        </p>
      )}
    </div>
  );
}
