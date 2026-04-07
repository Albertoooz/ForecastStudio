"use client";

import { useEffect, useState, useCallback } from "react";
import { AppShell } from "@/components/layout/AppShell";
import {
  Upload, FileSpreadsheet, Trash2, Settings2, Eye, Play,
  ChevronDown, ChevronUp, Loader2, CheckCircle2, X,
  TrendingUp, Calendar, Database, RefreshCw,
} from "lucide-react";
import { PostgresWizard } from "@/components/data/PostgresWizard";
import { connectionsApi, dataApi, modelsApi } from "@/lib/api";
import type { Dataset } from "@/types";

const FREQUENCIES = [
  { value: "D", label: "Daily" },
  { value: "W", label: "Weekly" },
  { value: "M", label: "Monthly" },
  { value: "Q", label: "Quarterly" },
  { value: "Y", label: "Yearly" },
  { value: "H", label: "Hourly" },
  { value: "T", label: "Minute" },
];

const MODEL_TYPES = [
  { value: "auto", label: "Auto (best model selection)" },
  { value: "prophet", label: "Prophet" },
  { value: "lightgbm", label: "LightGBM" },
  { value: "naive", label: "Naive Baseline" },
];

interface Preview {
  columns: string[];
  dtypes: Record<string, string>;
  head: Record<string, unknown>[];
  shape: [number, number];
  stats: Record<string, Record<string, number>>;
}

interface ConfigState {
  datetime_column: string;
  target_column: string;
  group_columns: string[];
  frequency: string;
}

interface TrainConfig {
  model_type: string;
  horizon: number;
  gap: number;
}

export default function DataPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploadingType, setUploadingType] = useState<"training" | "future_variables" | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);

  const [expanded, setExpanded] = useState<string | null>(null);
  const [preview, setPreview] = useState<Record<string, Preview>>({});
  const [previewLoading, setPreviewLoading] = useState<string | null>(null);

  const [configuring, setConfiguring] = useState<string | null>(null);
  const [configState, setConfigState] = useState<ConfigState>({
    datetime_column: "", target_column: "", group_columns: [], frequency: "",
  });
  const [linkedDatasetId, setLinkedDatasetId] = useState<string>("");
  const [configSaving, setConfigSaving] = useState(false);

  const [training, setTraining] = useState<string | null>(null);
  const [trainConfig, setTrainConfig] = useState<TrainConfig>({
    model_type: "auto", horizon: 12, gap: 0,
  });
  const [trainMsg, setTrainMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const [deleting, setDeleting] = useState<string | null>(null);
  const [pgWizardOpen, setPgWizardOpen] = useState(false);
  const [resyncingSourceId, setResyncingSourceId] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await dataApi.list();
      setDatasets(res.data.datasets || []);
    } catch { /* handled by interceptor */ }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    const refresh = () => {
      load();
    };

    const handleVisibility = () => {
      if (document.visibilityState === "visible") {
        load();
      }
    };

    window.addEventListener("datasets-updated", refresh as EventListener);
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", handleVisibility);

    return () => {
      window.removeEventListener("datasets-updated", refresh as EventListener);
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [load]);

  const trainingDatasets = datasets.filter((d) => d.dataset_type !== "future_variables");
  const futureDatasets = datasets.filter((d) => d.dataset_type === "future_variables");

  const handleUpload = async (
    e: React.ChangeEvent<HTMLInputElement>,
    type: "training" | "future_variables",
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";
    setUploadingType(type);
    setUploadError(null);
    try {
      const res = await dataApi.upload(file);
      const ds: Dataset = res.data;
      // Tag the dataset type immediately after upload
      if (type === "future_variables") {
        const patched = await dataApi.configureColumns(ds.id, { dataset_type: "future_variables" });
        setDatasets((prev) => [patched.data, ...prev]);
      } else {
        setDatasets((prev) => [ds, ...prev]);
      }
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        "Upload failed. Check file format.";
      setUploadError(msg);
    } finally {
      setUploadingType(null);
    }
  };

  const toggleExpand = async (ds: Dataset) => {
    if (expanded === ds.id) { setExpanded(null); return; }
    setExpanded(ds.id);
    if (!preview[ds.id]) {
      setPreviewLoading(ds.id);
      try {
        const res = await dataApi.preview(ds.id, 10);
        setPreview((p) => ({ ...p, [ds.id]: res.data }));
      } catch { /* noop */ }
      finally { setPreviewLoading(null); }
    }
  };

  const openConfig = (ds: Dataset) => {
    setConfiguring(ds.id);
    setConfigState({
      datetime_column: ds.datetime_column || "",
      target_column: ds.target_column || "",
      group_columns: ds.group_columns || [],
      frequency: ds.frequency || "",
    });
    setLinkedDatasetId(ds.linked_dataset_id || "");
  };

  const saveConfig = async (ds: Dataset) => {
    setConfigSaving(true);
    try {
      const res = await dataApi.configureColumns(ds.id, {
        datetime_column: configState.datetime_column || null,
        target_column: configState.target_column || null,
        group_columns: configState.group_columns.length ? configState.group_columns : null,
        frequency: configState.frequency || null,
        dataset_type: ds.dataset_type,
        linked_dataset_id: linkedDatasetId || null,
      });
      setDatasets((prev) => prev.map((d) => (d.id === ds.id ? res.data : d)));
      setConfiguring(null);
    } catch { /* noop */ }
    finally { setConfigSaving(false); }
  };

  const handleTrain = async (ds: Dataset) => {
    setTrainMsg(null);
    setTraining(ds.id);
    try {
      // Find linked future variables dataset
      const fv = futureDatasets.find((f) => f.linked_dataset_id === ds.id);
      const res = await modelsApi.train(ds.id, {
        model_type: trainConfig.model_type,
        horizon: trainConfig.horizon,
        gap: trainConfig.gap,
        config: {
          datetime_column: ds.datetime_column,
          target_column: ds.target_column,
          group_columns: ds.group_columns,
          frequency: ds.frequency,
          future_dataset_id: fv?.id,
        },
      });
      setTrainMsg({
        type: "success",
        text: `Training queued (ID: ${res.data.id.slice(0, 8)}…). Check Pipeline for progress.${fv ? ` Future variables: ${fv.name}` : ""}`,
      });
    } catch (err: unknown) {
      setTrainMsg({
        type: "error",
        text:
          (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
          "Training failed to start.",
      });
    } finally {
      setTraining(null);
    }
  };

  const handleDelete = async (ds: Dataset) => {
    if (!confirm(`Delete "${ds.name}"?`)) return;
    setDeleting(ds.id);
    try {
      await dataApi.delete(ds.id);
      setDatasets((prev) => prev.filter((d) => d.id !== ds.id));
    } catch { /* noop */ }
    finally { setDeleting(null); }
  };

  const handleResync = async (ds: Dataset) => {
    if (!ds.data_source_id) return;
    const st = (ds.source_type || "").toLowerCase();
    if (st !== "postgres" && st !== "sql") return;
    setResyncingSourceId(ds.data_source_id);
    setSyncError(null);
    try {
      const res = await connectionsApi.sync(ds.data_source_id);
      const updated = res.data as Dataset;
      setDatasets((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
      setPreview((p) => {
        const next = { ...p };
        delete next[ds.id];
        return next;
      });
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        "Resync failed";
      setSyncError(String(msg));
    } finally {
      setResyncingSourceId(null);
    }
  };

  const columns = (ds: Dataset): string[] => {
    if (preview[ds.id]) return preview[ds.id].columns;
    return ds.schema_columns || [];
  };

  return (
    <AppShell>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Data</h1>
            <p className="text-sm text-gray-500">
              Connect PostgreSQL once (training + optional future tables), or upload files. Snapshots
              go to blob storage for forecasting.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setPgWizardOpen(true)}
            className="flex shrink-0 items-center gap-2 rounded-xl border-2 border-brand-600 bg-white px-4 py-2 text-sm font-medium text-brand-700 hover:bg-brand-50 dark:border-brand-500 dark:bg-gray-900 dark:text-brand-300 dark:hover:bg-brand-950/40"
          >
            <Database className="h-4 w-4" />
            Connect PostgreSQL
          </button>
        </div>

        <PostgresWizard
          open={pgWizardOpen}
          onClose={() => setPgWizardOpen(false)}
          onCreated={(list) => setDatasets((prev) => [...list, ...prev])}
        />

        {/* Global messages */}
        {uploadError && (
          <div className="flex items-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <X className="h-4 w-4 shrink-0" />
            {uploadError}
            <button onClick={() => setUploadError(null)} className="ml-auto"><X className="h-3 w-3" /></button>
          </div>
        )}
        {syncError && (
          <div className="flex items-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
            <X className="h-4 w-4 shrink-0" />
            {syncError}
            <button onClick={() => setSyncError(null)} className="ml-auto"><X className="h-3 w-3" /></button>
          </div>
        )}
        {trainMsg && (
          <div
            className={`flex items-center gap-2 rounded-xl border px-4 py-3 text-sm ${
              trainMsg.type === "success"
                ? "border-green-200 bg-green-50 text-green-700"
                : "border-red-200 bg-red-50 text-red-700"
            }`}
          >
            {trainMsg.type === "success" ? (
              <CheckCircle2 className="h-4 w-4 shrink-0" />
            ) : (
              <X className="h-4 w-4 shrink-0" />
            )}
            {trainMsg.text}
            <button onClick={() => setTrainMsg(null)} className="ml-auto"><X className="h-3 w-3" /></button>
          </div>
        )}

        {/* ── TRAINING DATA ───────────────────────────────────────────────── */}
        <Section
          icon={<TrendingUp className="h-5 w-5 text-brand-600" />}
          title="Training Data"
          count={trainingDatasets.length}
          uploadLabel="Upload file"
          uploading={uploadingType === "training"}
          onUpload={(e) => handleUpload(e, "training")}
        >
          {loading ? (
            <EmptyState loading />
          ) : trainingDatasets.length === 0 ? (
            <EmptyState text="No training data yet — connect PostgreSQL or upload a file" />
          ) : (
            trainingDatasets.map((ds) => (
              <DatasetRow
                key={ds.id}
                ds={ds}
                expanded={expanded === ds.id}
                configuring={configuring === ds.id}
                previewLoading={previewLoading === ds.id}
                preview={preview[ds.id]}
                configState={configState}
                linkedDatasetId={linkedDatasetId}
                trainingDatasets={trainingDatasets}
                futureDatasets={futureDatasets}
                trainConfig={trainConfig}
                configSaving={configSaving}
                training={training === ds.id}
                deleting={deleting === ds.id}
                resyncing={
                  !!ds.data_source_id && resyncingSourceId === ds.data_source_id
                }
                columns={columns(ds)}
                onToggleExpand={() => toggleExpand(ds)}
                onOpenConfig={() => openConfig(ds)}
                onSaveConfig={() => saveConfig(ds)}
                onCancelConfig={() => setConfiguring(null)}
                onTrain={() => {
                  if (!ds.datetime_column || !ds.target_column) openConfig(ds);
                  else handleTrain(ds);
                }}
                onDelete={() => handleDelete(ds)}
                onResync={() => handleResync(ds)}
                onConfigChange={setConfigState}
                onLinkedChange={setLinkedDatasetId}
                onTrainConfigChange={setTrainConfig}
              />
            ))
          )}
        </Section>

        {/* ── FUTURE VARIABLES ───────────────────────────────────────────── */}
        <Section
          icon={<Calendar className="h-5 w-5 text-purple-600" />}
          title="Future Exogenous Variables"
          subtitle="Known future data used as regressors (e.g. promotions, holidays, prices)"
          count={futureDatasets.length}
          uploadLabel="Upload file"
          uploading={uploadingType === "future_variables"}
          onUpload={(e) => handleUpload(e, "future_variables")}
          accent="purple"
        >
          {loading ? (
            <EmptyState loading />
          ) : futureDatasets.length === 0 ? (
            <EmptyState text="No future variables yet — optional; connect Postgres or upload a file" />
          ) : (
            futureDatasets.map((ds) => (
              <DatasetRow
                key={ds.id}
                ds={ds}
                expanded={expanded === ds.id}
                configuring={configuring === ds.id}
                previewLoading={previewLoading === ds.id}
                preview={preview[ds.id]}
                configState={configState}
                linkedDatasetId={linkedDatasetId}
                trainingDatasets={trainingDatasets}
                futureDatasets={futureDatasets}
                trainConfig={trainConfig}
                configSaving={configSaving}
                training={false}
                deleting={deleting === ds.id}
                resyncing={
                  !!ds.data_source_id && resyncingSourceId === ds.data_source_id
                }
                columns={columns(ds)}
                onToggleExpand={() => toggleExpand(ds)}
                onOpenConfig={() => openConfig(ds)}
                onSaveConfig={() => saveConfig(ds)}
                onCancelConfig={() => setConfiguring(null)}
                onTrain={() => {}}
                onDelete={() => handleDelete(ds)}
                onResync={() => handleResync(ds)}
                onConfigChange={setConfigState}
                onLinkedChange={setLinkedDatasetId}
                onTrainConfigChange={setTrainConfig}
                isFutureVars
              />
            ))
          )}
        </Section>
      </div>
    </AppShell>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Section({
  icon,
  title,
  subtitle,
  count,
  uploadLabel,
  uploading,
  onUpload,
  accent = "brand",
  children,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle?: string;
  count: number;
  uploadLabel: string;
  uploading: boolean;
  onUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  accent?: "brand" | "purple";
  children: React.ReactNode;
}) {
  const btn =
    accent === "purple"
      ? "bg-purple-600 hover:bg-purple-700"
      : "bg-brand-600 hover:bg-brand-700";

  return (
    <div className="rounded-xl border bg-white shadow-sm dark:bg-gray-900 dark:border-gray-800">
      <div className="flex flex-col gap-3 border-b px-6 py-4 dark:border-gray-800 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          {icon}
          <div>
            <h2 className="font-semibold">
              {title} <span className="text-gray-400 font-normal">({count})</span>
            </h2>
            {subtitle && <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 sm:justify-end">
          <label
            className={`flex cursor-pointer items-center gap-2 rounded-xl ${btn} px-4 py-2 text-sm font-medium text-white transition-colors`}
          >
            {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            {uploading ? "Uploading…" : uploadLabel}
            <input
              type="file"
              accept=".csv,.xlsx,.xls,.parquet"
              onChange={onUpload}
              className="hidden"
              disabled={uploading}
            />
          </label>
        </div>
      </div>
      <div className="divide-y dark:divide-gray-800">{children}</div>
    </div>
  );
}

function EmptyState({ text, loading }: { text?: string; loading?: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center py-10 text-gray-400">
      {loading ? (
        <Loader2 className="h-6 w-6 animate-spin" />
      ) : (
        <>
          <FileSpreadsheet className="mb-3 h-10 w-10" />
          <p className="text-sm">{text}</p>
        </>
      )}
    </div>
  );
}

interface DatasetRowProps {
  ds: Dataset;
  expanded: boolean;
  configuring: boolean;
  previewLoading: boolean;
  preview?: Preview;
  configState: ConfigState;
  linkedDatasetId: string;
  trainingDatasets: Dataset[];
  futureDatasets: Dataset[];
  trainConfig: TrainConfig;
  configSaving: boolean;
  training: boolean;
  deleting: boolean;
  resyncing?: boolean;
  columns: string[];
  isFutureVars?: boolean;
  onToggleExpand: () => void;
  onOpenConfig: () => void;
  onSaveConfig: () => void;
  onCancelConfig: () => void;
  onTrain: () => void;
  onDelete: () => void;
  onResync?: () => void;
  onConfigChange: (s: ConfigState) => void;
  onLinkedChange: (id: string) => void;
  onTrainConfigChange: (c: TrainConfig) => void;
}

function DatasetRow({
  ds, expanded, configuring, previewLoading, preview, configState, linkedDatasetId,
  trainingDatasets, futureDatasets, trainConfig, configSaving, training, deleting,
  resyncing = false,
  columns, isFutureVars = false,
  onToggleExpand, onOpenConfig, onSaveConfig, onCancelConfig, onTrain, onDelete, onResync,
  onConfigChange, onLinkedChange, onTrainConfigChange,
}: DatasetRowProps) {
  const linkedFV = !isFutureVars
    ? futureDatasets.find((f) => f.linked_dataset_id === ds.id)
    : null;

  const isConfigured = !!ds.datetime_column && (isFutureVars || !!ds.target_column);
  const src = (ds.source_type || "").toLowerCase();
  const canResync =
    !!ds.data_source_id && (src === "postgres" || src === "sql") && !!onResync;

  const sourceBadge =
    src === "postgres" || src === "sql" ? (
      <span className="ml-2 shrink-0 rounded-md bg-sky-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-sky-800 dark:bg-sky-950/50 dark:text-sky-200">
        PostgreSQL
      </span>
    ) : src === "file" ? (
      <span className="ml-2 shrink-0 rounded-md bg-gray-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-gray-700 dark:bg-gray-800 dark:text-gray-300">
        File
      </span>
    ) : null;

  return (
    <div>
      {/* Main row */}
      <div className="flex items-center justify-between px-6 py-4 hover:bg-gray-50 dark:hover:bg-gray-800/50">
        <div className="min-w-0 flex-1">
          <p className="flex flex-wrap items-center truncate font-medium">
            <span className="truncate">{ds.name}</span>
            {sourceBadge}
            {ds.sync_status === "syncing" && (
              <span className="ml-2 text-[10px] text-amber-600">syncing…</span>
            )}
          </p>
          <p className="text-xs text-gray-500">
            {ds.rows != null ? `${ds.rows.toLocaleString()} rows` : "?"}
            {ds.columns != null ? ` × ${ds.columns} cols` : ""}
            {ds.frequency && ` • freq: ${ds.frequency}`}
            {ds.datetime_column && ` • date: ${ds.datetime_column}`}
            {ds.target_column && ` • target: ${ds.target_column}`}
            {linkedFV && (
              <span className="ml-1 text-purple-500">• FV: {linkedFV.name}</span>
            )}
            {ds.last_sync_at && (
              <span className="ml-1 text-gray-400">
                • synced {new Date(ds.last_sync_at).toLocaleString()}
              </span>
            )}
          </p>
          {ds.sync_status === "error" && ds.last_error && (
            <p className="mt-1 text-xs text-red-600 dark:text-red-400">{ds.last_error}</p>
          )}
        </div>

        <div className="ml-4 flex shrink-0 items-center gap-1">
          {canResync && (
            <button
              type="button"
              onClick={onResync}
              disabled={resyncing}
              title="Re-sync snapshot from database"
              className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-brand-600 dark:hover:bg-gray-700 disabled:opacity-40"
            >
              <RefreshCw className={`h-4 w-4 ${resyncing ? "animate-spin" : ""}`} />
            </button>
          )}
          <button
            onClick={onToggleExpand}
            title="Preview data"
            className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <Eye className="h-4 w-4" />
          </button>
          <button
            onClick={onOpenConfig}
            title="Configure columns"
            className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <Settings2 className="h-4 w-4" />
          </button>

          {!isFutureVars && (
            <button
              onClick={onTrain}
              disabled={training}
              title={!isConfigured ? "Configure columns first" : "Start training"}
              className={`flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors disabled:opacity-40 ${
                !isConfigured
                  ? "bg-yellow-500 text-white hover:bg-yellow-600"
                  : "bg-brand-600 text-white hover:bg-brand-700"
              }`}
            >
              {training ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : !isConfigured ? (
                <Settings2 className="h-3 w-3" />
              ) : (
                <Play className="h-3 w-3" />
              )}
              {!isConfigured ? "Configure" : "Train"}
            </button>
          )}

          <button
            onClick={onDelete}
            disabled={deleting}
            className="rounded-lg p-2 text-gray-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-900/20 disabled:opacity-40"
          >
            {deleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
          </button>

          <button
            onClick={onToggleExpand}
            className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* Config panel */}
      {configuring && (
        <div className="border-t bg-gray-50 px-6 py-5 dark:border-gray-800 dark:bg-gray-800/50">
          <h3 className="mb-4 text-sm font-semibold">Column Configuration</h3>
          <div className={`grid gap-4 ${isFutureVars ? "sm:grid-cols-2" : "sm:grid-cols-2 lg:grid-cols-4"}`}>
            <label className="space-y-1">
              <span className="text-xs font-medium text-gray-600 dark:text-gray-400">Date / Time column</span>
              <select
                value={configState.datetime_column}
                onChange={(e) => onConfigChange({ ...configState, datetime_column: e.target.value })}
                className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
              >
                <option value="">— select —</option>
                {columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>

            {!isFutureVars && (
              <label className="space-y-1">
                <span className="text-xs font-medium text-gray-600 dark:text-gray-400">Target column (y)</span>
                <select
                  value={configState.target_column}
                  onChange={(e) => onConfigChange({ ...configState, target_column: e.target.value })}
                  className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
                >
                  <option value="">— select —</option>
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
            )}

            <label className="space-y-1">
              <span className="text-xs font-medium text-gray-600 dark:text-gray-400">Frequency</span>
              <select
                value={configState.frequency}
                onChange={(e) => onConfigChange({ ...configState, frequency: e.target.value })}
                className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
              >
                <option value="">— auto-detect —</option>
                {FREQUENCIES.map((f) => <option key={f.value} value={f.value}>{f.label} ({f.value})</option>)}
              </select>
            </label>

            {!isFutureVars && (
              <label className="space-y-1">
                <span className="text-xs font-medium text-gray-600 dark:text-gray-400">Group column(s)</span>
                <select
                  multiple
                  value={configState.group_columns}
                  onChange={(e) => {
                    const vals = Array.from(e.target.selectedOptions, (o) => o.value);
                    onConfigChange({ ...configState, group_columns: vals });
                  }}
                  className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
                  size={3}
                >
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
                <p className="text-xs text-gray-400">Ctrl+click for multiple</p>
              </label>
            )}

            {isFutureVars && (
              <label className="space-y-1">
                <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                  Link to training dataset
                </span>
                <select
                  value={linkedDatasetId}
                  onChange={(e) => onLinkedChange(e.target.value)}
                  className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
                >
                  <option value="">— select training dataset —</option>
                  {trainingDatasets.map((t) => (
                    <option key={t.id} value={t.id}>{t.name}</option>
                  ))}
                </select>
              </label>
            )}
          </div>

          {/* Training config (only for training datasets) */}
          {!isFutureVars && (
            <div className="mt-4 grid gap-4 sm:grid-cols-3">
              <label className="space-y-1">
                <span className="text-xs font-medium text-gray-600 dark:text-gray-400">Model type</span>
                <select
                  value={trainConfig.model_type}
                  onChange={(e) => onTrainConfigChange({ ...trainConfig, model_type: e.target.value })}
                  className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
                >
                  {MODEL_TYPES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
                </select>
              </label>
              <label className="space-y-1">
                <span className="text-xs font-medium text-gray-600 dark:text-gray-400">Horizon (periods)</span>
                <input
                  type="number" min={1} max={365} value={trainConfig.horizon}
                  onChange={(e) => onTrainConfigChange({ ...trainConfig, horizon: Number(e.target.value) })}
                  className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
                />
              </label>
              <label className="space-y-1">
                <span className="text-xs font-medium text-gray-600 dark:text-gray-400">Gap (periods)</span>
                <input
                  type="number" min={0} max={100} value={trainConfig.gap}
                  onChange={(e) => onTrainConfigChange({ ...trainConfig, gap: Number(e.target.value) })}
                  className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700"
                />
              </label>
            </div>
          )}

          <div className="mt-4 flex gap-2">
            <button
              onClick={onSaveConfig}
              disabled={configSaving}
              className="flex items-center gap-1 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
            >
              {configSaving && <Loader2 className="h-3 w-3 animate-spin" />}
              Save
            </button>
            <button
              onClick={onCancelConfig}
              className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Preview panel */}
      {expanded && (
        <div className="border-t dark:border-gray-800">
          {previewLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-gray-400" />
            </div>
          ) : preview ? (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead className="bg-gray-50 dark:bg-gray-800">
                  <tr>
                    {preview.columns.map((col) => (
                      <th key={col} className="px-4 py-2 text-left font-medium text-gray-500">
                        {col}
                        <span className="ml-1 text-gray-300">{preview.dtypes[col]}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y dark:divide-gray-800">
                  {preview.head.map((row, i) => (
                    <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                      {preview.columns.map((col) => (
                        <td key={col} className="px-4 py-2 text-gray-700 dark:text-gray-300">
                          {String(row[col] ?? "")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="px-4 py-2 text-xs text-gray-400">
                Showing {preview.head.length} of {ds.rows?.toLocaleString()} rows
              </p>
            </div>
          ) : (
            <p className="px-6 py-4 text-sm text-gray-500">Preview not available.</p>
          )}
        </div>
      )}
    </div>
  );
}
