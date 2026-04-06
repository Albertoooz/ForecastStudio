"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { AppShell } from "@/components/layout/AppShell";
import {
  GitBranch, CheckCircle2, Clock, XCircle, Loader2,
  RefreshCw, ChevronDown, ChevronUp,
} from "lucide-react";
import { modelsApi } from "@/lib/api";
import type { ModelRun, PipelineStep } from "@/types";

const STATUS_ICON = {
  queued: <Clock className="h-4 w-4 text-gray-400" />,
  running: <Loader2 className="h-4 w-4 animate-spin text-brand-500" />,
  completed: <CheckCircle2 className="h-4 w-4 text-green-500" />,
  failed: <XCircle className="h-4 w-4 text-red-500" />,
} as const;

const STATUS_BADGE: Record<string, string> = {
  queued: "bg-gray-50 text-gray-500 dark:bg-gray-800",
  running: "bg-brand-50 text-brand-700 dark:bg-brand-900/20 dark:text-brand-400",
  completed: "bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400",
  failed: "bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-400",
};

const STEP_LABELS: Record<string, string> = {
  analysis: "Data Analysis",
  data_analysis: "Data Analysis",
  preparation: "Data Preparation",
  features: "Feature Engineering",
  feature_engineering: "Feature Engineering",
  model_selection: "Model Selection",
  model_training: "Training",
  training: "Training",
  evaluation: "Evaluation",
  forecast: "Forecast Generation",
  error: "Error",
};

export default function PipelinePage() {
  const [runs, setRuns] = useState<ModelRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [steps, setSteps] = useState<Record<string, PipelineStep[]>>({});
  const [stepsLoading, setStepsLoading] = useState<string | null>(null);

  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  const loadRuns = useCallback(async (silent = false) => {
    if (!silent) setRefreshing(true);
    try {
      const res = await modelsApi.listRuns({ limit: 20 });
      setRuns(res.data.runs ?? []);
    } catch {
      /* handled by interceptor */
    } finally {
      if (!silent) setRefreshing(false);
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  // Auto-refresh while any run is in running/queued state
  useEffect(() => {
    const hasActive = runs.some((r) => r.status === "running" || r.status === "queued");
    if (hasActive) {
      pollingRef.current = setInterval(() => loadRuns(true), 3000);
    } else {
      if (pollingRef.current) clearInterval(pollingRef.current);
    }
    return () => { if (pollingRef.current) clearInterval(pollingRef.current); };
  }, [runs, loadRuns]);

  const toggleExpand = async (run: ModelRun) => {
    if (expanded === run.id) {
      setExpanded(null);
      return;
    }
    setExpanded(run.id);

    if (!steps[run.id]) {
      setStepsLoading(run.id);
      try {
        const res = await modelsApi.getPipeline(run.id);
        setSteps((p) => ({ ...p, [run.id]: res.data }));
      } catch {
        /* noop */
      } finally {
        setStepsLoading(null);
      }
    }
  };

  const handlePromote = async (run: ModelRun) => {
    try {
      await modelsApi.promote(run.id);
      await loadRuns();
    } catch {
      /* noop */
    }
  };

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Pipeline</h1>
            <p className="text-sm text-gray-500">
              Agent pipeline runs — data analysis → feature engineering → model training
            </p>
          </div>
          <button
            onClick={() => loadRuns()}
            disabled={refreshing}
            className="flex items-center gap-2 rounded-xl border px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:border-gray-700 dark:hover:bg-gray-800 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>

        {/* Run list */}
        <div className="rounded-xl border bg-white shadow-sm dark:bg-gray-900 dark:border-gray-800">
          <div className="border-b px-6 py-4 dark:border-gray-800">
            <h2 className="font-semibold">Run History ({runs.length})</h2>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-6 w-6 animate-spin text-gray-300" />
            </div>
          ) : runs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-gray-400">
              <GitBranch className="mb-4 h-12 w-12" />
              <p className="text-sm">No runs yet — upload a dataset and click Train</p>
            </div>
          ) : (
            <div className="divide-y dark:divide-gray-800">
              {runs.map((run) => (
                <div key={run.id}>
                  <div className="flex items-center gap-4 px-6 py-4 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                    <div className="flex-shrink-0">
                      {STATUS_ICON[run.status as keyof typeof STATUS_ICON] ??
                        STATUS_ICON.queued}
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <p className="truncate font-mono text-xs text-gray-400">
                          {run.id.slice(0, 8)}…
                        </p>
                        <span
                          className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                            STATUS_BADGE[run.status] ?? STATUS_BADGE.queued
                          }`}
                        >
                          {run.status}
                        </span>
                      </div>
                      <p className="text-sm font-medium">
                        {run.best_model_name || run.model_type}
                        {" "}
                        <span className="text-gray-400 font-normal">
                          • horizon {run.horizon} • gap {run.gap}
                        </span>
                      </p>
                      {run.metrics && (
                        <p className="text-xs text-gray-400 mt-0.5">
                          RMSE {run.metrics.rmse?.toFixed(3)} •
                          MAPE {run.metrics.mape?.toFixed(2)}% •
                          Improvement {run.metrics.rmse_improvement_pct?.toFixed(1)}%
                        </p>
                      )}
                    </div>

                    <div className="flex items-center gap-2 text-xs text-gray-400">
                      {run.duration_seconds != null && (
                        <span>{run.duration_seconds.toFixed(0)}s</span>
                      )}
                      <span>{new Date(run.created_at).toLocaleString()}</span>
                      {run.status === "completed" && (
                        <button
                          onClick={() => handlePromote(run)}
                          className="rounded-lg border border-brand-300 px-3 py-1 text-xs font-medium text-brand-600 hover:bg-brand-50 dark:border-brand-700 dark:text-brand-400 dark:hover:bg-brand-900/20"
                        >
                          Promote
                        </button>
                      )}
                      <button
                        onClick={() => toggleExpand(run)}
                        className="rounded-lg p-1 hover:bg-gray-100 dark:hover:bg-gray-700"
                      >
                        {expanded === run.id ? (
                          <ChevronUp className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Pipeline steps */}
                  {expanded === run.id && (
                    <div className="border-t bg-gray-50 px-8 py-5 dark:border-gray-800 dark:bg-gray-800/30">
                      <h3 className="mb-4 text-sm font-semibold text-gray-600 dark:text-gray-400">
                        Pipeline Steps
                      </h3>
                      {stepsLoading === run.id ? (
                        <div className="flex items-center gap-2 text-sm text-gray-400">
                          <Loader2 className="h-4 w-4 animate-spin" /> Loading…
                        </div>
                      ) : steps[run.id]?.length ? (
                        <div className="space-y-2">
                          {steps[run.id].map((step, i) => (
                            <div key={i} className="flex items-start gap-3">
                              <div className="mt-0.5 flex-shrink-0">
                                {STATUS_ICON[step.status as keyof typeof STATUS_ICON] ??
                                  STATUS_ICON.queued}
                              </div>
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center gap-2">
                                  <p className="text-sm font-medium">
                                    {STEP_LABELS[step.step_name] || step.step_name}
                                  </p>
                                  {step.agent_name && (
                                    <span className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-xs text-gray-500 dark:bg-gray-700">
                                      {step.agent_name}
                                    </span>
                                  )}
                                  {step.duration_seconds != null && (
                                    <span className="text-xs text-gray-400">
                                      {step.duration_seconds.toFixed(1)}s
                                    </span>
                                  )}
                                </div>
                                {step.message && (
                                  <p className="mt-0.5 text-xs text-gray-500">{step.message}</p>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm text-gray-500">No detailed steps recorded yet.</p>
                      )}

                      {/* Show metrics if completed */}
                      {run.status === "completed" && run.metrics && (
                        <div className="mt-4 grid grid-cols-3 gap-3">
                          {Object.entries(run.metrics).map(([k, v]) => (
                            <div key={k} className="rounded-lg bg-white p-3 text-center shadow-sm dark:bg-gray-900">
                              <p className="text-xs text-gray-400 uppercase">{k.replace(/_/g, " ")}</p>
                              <p className="mt-1 font-mono text-sm font-semibold">
                                {typeof v === "number" ? v.toFixed(3) : String(v)}
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
