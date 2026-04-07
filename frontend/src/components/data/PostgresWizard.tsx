"use client";

import { useState } from "react";
import { X, Loader2, Database } from "lucide-react";
import { connectionsApi, dataApi } from "@/lib/api";
import type { Dataset } from "@/types";

export function PostgresWizard({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (datasets: Dataset[]) => void;
}) {
  const [step, setStep] = useState<1 | 2>(1);
  const [host, setHost] = useState("localhost");
  const [port, setPort] = useState("5432");
  const [database, setDatabase] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const [tables, setTables] = useState<string[]>([]);
  const [trainTable, setTrainTable] = useState("");
  const [trainSql, setTrainSql] = useState("");
  const [trainName, setTrainName] = useState("");

  const [addFuture, setAddFuture] = useState(false);
  const [fvTable, setFvTable] = useState("");
  const [fvSql, setFvSql] = useState("");
  const [fvName, setFvName] = useState("");

  const [testing, setTesting] = useState(false);
  const [loadingTables, setLoadingTables] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const baseConn = () => ({
    host,
    port: parseInt(port, 10) || 5432,
    database,
    username,
    password,
  });

  const pgPayload = (table: string, sql: string) => {
    const q = sql.trim();
    return {
      ...baseConn(),
      query: q ? q : null,
      table: q ? null : table || null,
    };
  };

  const reset = () => {
    setStep(1);
    setTables([]);
    setTrainTable("");
    setTrainSql("");
    setTrainName("");
    setAddFuture(false);
    setFvTable("");
    setFvSql("");
    setFvName("");
    setError(null);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  if (!open) return null;

  const testConn = async () => {
    setError(null);
    setTesting(true);
    try {
      await connectionsApi.test(pgPayload("", ""));
      setStep(2);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        "Connection failed";
      setError(String(msg));
    } finally {
      setTesting(false);
    }
  };

  const loadTables = async () => {
    setError(null);
    setLoadingTables(true);
    try {
      const res = await connectionsApi.probeTables(pgPayload("", ""));
      setTables(res.data.tables || []);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        "Could not list tables";
      setError(String(msg));
    } finally {
      setLoadingTables(false);
    }
  };

  const createDatasets = async () => {
    setError(null);
    if (!trainName.trim()) {
      setError("Enter a name for the training dataset");
      return;
    }
    if (!trainSql.trim() && !trainTable) {
      setError("Choose a training table or enter SQL for training data");
      return;
    }
    if (addFuture) {
      if (!fvName.trim()) {
        setError("Enter a name for the future variables dataset");
        return;
      }
      if (!fvSql.trim() && !fvTable) {
        setError("Choose a table or SQL for future variables");
        return;
      }
    }

    setCreating(true);
    try {
      const trainRes = await connectionsApi.create({
        name: trainName.trim(),
        source_type: "postgres",
        postgres: pgPayload(trainTable, trainSql),
        dataset_type: "training",
      });
      const trainingDs = trainRes.data as Dataset;
      const created: Dataset[] = [trainingDs];

      if (addFuture) {
        const fvRes = await connectionsApi.create({
          name: fvName.trim(),
          source_type: "postgres",
          postgres: pgPayload(fvTable, fvSql),
          dataset_type: "future_variables",
        });
        let fvDs = fvRes.data as Dataset;
        const linked = await dataApi.configureColumns(fvDs.id, {
          dataset_type: "future_variables",
          linked_dataset_id: trainingDs.id,
        });
        fvDs = linked.data as Dataset;
        created.push(fvDs);
      }

      onCreated(created);
      handleClose();
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        "Create failed";
      setError(String(msg));
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-2xl border bg-white shadow-xl dark:border-gray-700 dark:bg-gray-900">
        <div className="flex items-center justify-between border-b px-5 py-4 dark:border-gray-800">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-brand-600" />
            <div>
              <h2 className="font-semibold">Connect PostgreSQL</h2>
              <p className="text-xs text-gray-500">
                One connection — then map training data and optional future variables
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleClose}
            className="rounded-lg p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-4 px-5 py-4">
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
              {error}
            </div>
          )}

          {step === 1 && (
            <>
              <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-200">
                <strong>Backend w Dockerze:</strong> zamiast <code className="rounded bg-white/80 px-1 dark:bg-gray-800">localhost</code>{" "}
                użyj <code className="rounded bg-white/80 px-1 dark:bg-gray-800">host.docker.internal</code>, żeby
                dotrzeć do Postgresa na Twoim komputerze (np. port <strong>5434</strong>). Gdy API działa bez Dockera,
                wtedy <code className="rounded bg-white/80 px-1">localhost</code> jest OK.
              </p>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">
                  Host
                  <input
                    value={host}
                    onChange={(e) => setHost(e.target.value)}
                    placeholder="host.docker.internal"
                    className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-800"
                  />
                </label>
                <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">
                  Port
                  <input
                    value={port}
                    onChange={(e) => setPort(e.target.value)}
                    placeholder="5432"
                    className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-800"
                  />
                </label>
                <label className="col-span-full block text-xs font-medium text-gray-600 dark:text-gray-400">
                  Database
                  <input
                    value={database}
                    onChange={(e) => setDatabase(e.target.value)}
                    placeholder="mockdb"
                    className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-800"
                  />
                </label>
                <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">
                  Username
                  <input
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    autoComplete="off"
                    className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-800"
                  />
                </label>
                <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">
                  Password
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    autoComplete="new-password"
                    className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-800"
                  />
                </label>
              </div>
              <div className="flex justify-end border-t pt-3 dark:border-gray-800">
                <button
                  type="button"
                  onClick={testConn}
                  disabled={testing || !database || !username}
                  className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-40"
                >
                  {testing ? <Loader2 className="h-4 w-4 animate-spin" /> : "Test connection & continue"}
                </button>
              </div>
            </>
          )}

          {step === 2 && (
            <>
              <button
                type="button"
                onClick={() => {
                  setStep(1);
                  setError(null);
                }}
                className="text-sm text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              >
                ← Back to credentials
              </button>

              <button
                type="button"
                onClick={loadTables}
                disabled={loadingTables}
                className="w-full rounded-lg border border-brand-600 py-2 text-sm font-medium text-brand-600 hover:bg-brand-50 dark:hover:bg-brand-950/30 disabled:opacity-50"
              >
                {loadingTables ? (
                  <Loader2 className="mx-auto h-4 w-4 animate-spin" />
                ) : (
                  "Load table list from database"
                )}
              </button>

              <div className="rounded-xl border border-brand-200 bg-brand-50/50 p-4 dark:border-brand-900 dark:bg-brand-950/20">
                <h3 className="mb-2 text-sm font-semibold text-brand-800 dark:text-brand-200">
                  Training data <span className="text-red-600">*</span>
                </h3>
                {tables.length > 0 && (
                  <label className="mb-2 block text-xs font-medium text-gray-600 dark:text-gray-400">
                    Table
                    <select
                      value={trainTable}
                      onChange={(e) => {
                        setTrainTable(e.target.value);
                        setTrainSql("");
                      }}
                      className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-800"
                    >
                      <option value="">— select schema.table —</option>
                      {tables.map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">
                  Or custom SQL (overrides table)
                  <textarea
                    value={trainSql}
                    onChange={(e) => {
                      setTrainSql(e.target.value);
                      if (e.target.value.trim()) setTrainTable("");
                    }}
                    rows={3}
                    placeholder="SELECT * FROM public.sales ..."
                    className="mt-1 w-full rounded-lg border px-3 py-2 font-mono text-xs dark:border-gray-700 dark:bg-gray-800"
                  />
                </label>
                <label className="mt-2 block text-xs font-medium text-gray-600 dark:text-gray-400">
                  Dataset name
                  <input
                    value={trainName}
                    onChange={(e) => setTrainName(e.target.value)}
                    placeholder="e.g. Sales training"
                    className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-800"
                  />
                </label>
              </div>

              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={addFuture}
                  onChange={(e) => setAddFuture(e.target.checked)}
                  className="rounded border-gray-300"
                />
                Also add future exogenous variables (second dataset, same DB connection)
              </label>

              {addFuture && (
                <div className="rounded-xl border border-purple-200 bg-purple-50/50 p-4 dark:border-purple-900 dark:bg-purple-950/20">
                  <h3 className="mb-2 text-sm font-semibold text-purple-800 dark:text-purple-200">
                    Future variables
                  </h3>
                  {tables.length > 0 && (
                    <label className="mb-2 block text-xs font-medium text-gray-600 dark:text-gray-400">
                      Table
                      <select
                        value={fvTable}
                        onChange={(e) => {
                          setFvTable(e.target.value);
                          setFvSql("");
                        }}
                        className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-800"
                      >
                        <option value="">— select —</option>
                        {tables.map((t) => (
                          <option key={t} value={t}>
                            {t}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                  <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">
                    Or custom SQL
                    <textarea
                      value={fvSql}
                      onChange={(e) => {
                        setFvSql(e.target.value);
                        if (e.target.value.trim()) setFvTable("");
                      }}
                      rows={3}
                      className="mt-1 w-full rounded-lg border px-3 py-2 font-mono text-xs dark:border-gray-700 dark:bg-gray-800"
                    />
                  </label>
                  <label className="mt-2 block text-xs font-medium text-gray-600 dark:text-gray-400">
                    Dataset name
                    <input
                      value={fvName}
                      onChange={(e) => setFvName(e.target.value)}
                      placeholder="e.g. Promotions future"
                      className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-800"
                    />
                  </label>
                </div>
              )}

              <button
                type="button"
                onClick={createDatasets}
                disabled={creating}
                className="w-full rounded-lg bg-brand-600 py-2.5 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50"
              >
                {creating ? (
                  <Loader2 className="mx-auto h-4 w-4 animate-spin" />
                ) : addFuture ? (
                  "Create training + future datasets"
                ) : (
                  "Create training dataset"
                )}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
