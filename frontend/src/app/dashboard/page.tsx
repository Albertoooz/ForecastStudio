"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import {
  BarChart3, Database, GitBranch, AlertTriangle,
  CheckCircle2, XCircle, Loader2, Clock,
} from "lucide-react";
import { dataApi, modelsApi, monitoringApi } from "@/lib/api";
import type { ModelRun, MonitoringSummary } from "@/types";

function StatCard({
  title,
  value,
  icon: Icon,
  color = "brand",
  loading,
}: {
  title: string;
  value: string | number;
  icon: React.ElementType;
  color?: "brand" | "green" | "red" | "yellow";
  loading?: boolean;
}) {
  const colorMap = {
    brand: "bg-brand-50 text-brand-600 dark:bg-brand-900/20",
    green: "bg-green-50 text-green-600 dark:bg-green-900/20",
    red: "bg-red-50 text-red-600 dark:bg-red-900/20",
    yellow: "bg-yellow-50 text-yellow-600 dark:bg-yellow-900/20",
  };

  return (
    <div className="rounded-xl border bg-white p-6 shadow-sm dark:bg-gray-900 dark:border-gray-800">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
          <p className="mt-2 text-3xl font-bold">
            {loading ? <Loader2 className="h-6 w-6 animate-spin text-gray-300" /> : value}
          </p>
        </div>
        <div className={`rounded-lg p-3 ${colorMap[color]}`}>
          <Icon className="h-6 w-6" />
        </div>
      </div>
    </div>
  );
}

const runStatus = {
  queued: { icon: Clock, cls: "text-gray-500 bg-gray-50", label: "queued" },
  running: { icon: Loader2, cls: "text-brand-600 bg-brand-50", label: "running" },
  completed: { icon: CheckCircle2, cls: "text-green-600 bg-green-50", label: "completed" },
  failed: { icon: XCircle, cls: "text-red-600 bg-red-50", label: "failed" },
} as const;

function RunBadge({ status }: { status: string }) {
  const cfg = runStatus[status as keyof typeof runStatus] ?? runStatus.queued;
  const Icon = cfg.icon;
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${cfg.cls}`}
    >
      <Icon className={`h-3 w-3 ${status === "running" ? "animate-spin" : ""}`} />
      {cfg.label}
    </span>
  );
}

export default function DashboardPage() {
  const [datasetCount, setDatasetCount] = useState<number | null>(null);
  const [runs, setRuns] = useState<ModelRun[]>([]);
  const [monitoring, setMonitoring] = useState<MonitoringSummary | null>(null);
  const [fetching, setFetching] = useState(true);

  useEffect(() => {
    const go = async () => {
      try {
        const [dsRes, runsRes, monRes] = await Promise.allSettled([
          dataApi.list(1, 0),
          modelsApi.listRuns({ limit: 8 }),
          monitoringApi.summary(),
        ]);

        if (dsRes.status === "fulfilled") setDatasetCount(dsRes.value.data.total ?? 0);
        if (runsRes.status === "fulfilled") setRuns(runsRes.value.data.runs ?? []);
        if (monRes.status === "fulfilled") setMonitoring(monRes.value.data);
      } finally {
        setFetching(false);
      }
    };
    go();
  }, []);

  const completedRuns = runs.filter((r) => r.status === "completed");
  const totalRuns = runs.length;

  return (
    <AppShell>
      <div className="space-y-8">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-gray-500 dark:text-gray-400">Forecasting platform overview</p>
        </div>

        {/* Stats */}
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title="Datasets"
            value={datasetCount ?? "—"}
            icon={Database}
            color="brand"
            loading={fetching}
          />
          <StatCard
            title="Active Models"
            value={monitoring?.active_versions ?? "—"}
            icon={BarChart3}
            color="green"
            loading={fetching}
          />
          <StatCard
            title="Pipeline Runs"
            value={totalRuns}
            icon={GitBranch}
            color="brand"
            loading={fetching}
          />
          <StatCard
            title="Alerts (24h)"
            value={monitoring?.total_alerts_24h ?? "—"}
            icon={AlertTriangle}
            color={monitoring && monitoring.total_alerts_24h > 0 ? "red" : "brand"}
            loading={fetching}
          />
        </div>

        {/* Recent activity */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Recent runs */}
          <div className="rounded-xl border bg-white p-6 shadow-sm dark:bg-gray-900 dark:border-gray-800">
            <h2 className="mb-4 text-lg font-semibold">Recent Training Runs</h2>
            {fetching ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-5 w-5 animate-spin text-gray-300" />
              </div>
            ) : runs.length === 0 ? (
              <p className="text-sm text-gray-500">No runs yet — upload a dataset and click Train</p>
            ) : (
              <div className="space-y-3">
                {runs.slice(0, 6).map((r) => (
                  <div key={r.id} className="flex items-center justify-between">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">{r.best_model_name || r.model_type}</p>
                      <p className="text-xs text-gray-400">
                        Horizon {r.horizon} • {new Date(r.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <RunBadge status={r.status} />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Best metrics from completed runs */}
          <div className="rounded-xl border bg-white p-6 shadow-sm dark:bg-gray-900 dark:border-gray-800">
            <h2 className="mb-4 text-lg font-semibold">Model Metrics</h2>
            {fetching ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-5 w-5 animate-spin text-gray-300" />
              </div>
            ) : completedRuns.length === 0 ? (
              <p className="text-sm text-gray-500">No completed runs yet</p>
            ) : (
              <div className="space-y-4">
                {completedRuns.slice(0, 4).map((r) => (
                  <div key={r.id} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">{r.best_model_name || r.model_type}</span>
                      {r.metrics?.health_score != null && (
                        <span className="text-xs text-gray-500">
                          Health: {r.metrics.health_score}/100
                        </span>
                      )}
                    </div>
                    {r.metrics && (
                      <div className="grid grid-cols-3 gap-2">
                        {["rmse", "mape", "rmse_improvement_pct"].map((k) => (
                          r.metrics![k] != null && (
                            <div key={k} className="rounded-lg bg-gray-50 p-2 text-center dark:bg-gray-800">
                              <p className="text-xs text-gray-400">{k.toUpperCase().replace("_PCT", " %")}</p>
                              <p className="font-mono text-sm font-semibold">
                                {Number(r.metrics![k]).toFixed(2)}
                                {k.includes("pct") ? "%" : ""}
                              </p>
                            </div>
                          )
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
