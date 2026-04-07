/**
 * API client for the forecaster backend.
 */

import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: `${API_BASE}/api`,
  headers: { "Content-Type": "application/json" },
});

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

// Handle 401 → redirect to login (except when already on login page, so error can be shown)
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401 && typeof window !== "undefined") {
      const isLoginPage = window.location.pathname === "/login";
      if (!isLoginPage) {
        localStorage.removeItem("access_token");
        window.location.href = "/login";
      }
    }
    return Promise.reject(err);
  }
);

// ── Auth ─────────────────────────────────────────────────────────────────────

export const authApi = {
  login: (email: string, password: string) =>
    api.post("/auth/login", new URLSearchParams({ username: email, password }), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    }),
  register: (email: string, password: string, fullName: string) =>
    api.post("/auth/register", { email, password, full_name: fullName }),
  me: () => api.get("/auth/me"),
};

// ── Data ─────────────────────────────────────────────────────────────────────

export const dataApi = {
  upload: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/data/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  list: (limit = 50, offset = 0) =>
    api.get("/data/list", { params: { limit, offset } }),
  preview: (datasetId: string, rows = 20) =>
    api.get(`/data/${datasetId}/preview`, { params: { rows } }),
  configureColumns: (
    datasetId: string,
    config: {
      datetime_column?: string | null;
      target_column?: string | null;
      group_columns?: string[] | null;
      frequency?: string | null;
      dataset_type?: string | null;
      linked_dataset_id?: string | null;
    }
  ) => api.patch(`/data/${datasetId}/columns`, config),
  transform: (datasetId: string, operation: string, params: Record<string, unknown>) =>
    api.post(`/data/${datasetId}/transform`, { operation, params }),
  delete: (datasetId: string) => api.delete(`/data/${datasetId}`),
};

// ── Data connections (Postgres → snapshot in blob) ───────────────────────────

export const connectionsApi = {
  test: (postgres: Record<string, unknown>) =>
    api.post("/data/connections/test", { source_type: "postgres", postgres }),
  probeTables: (postgres: Record<string, unknown>) =>
    api.post("/data/connections/probe/tables", { source_type: "postgres", postgres }),
  probePreview: (
    postgres: Record<string, unknown>,
    table: string,
    rows = 20,
  ) =>
    api.post("/data/connections/probe/preview", {
      source_type: "postgres",
      postgres,
      table,
      rows,
    }),
  create: (body: {
    name: string;
    source_type: string;
    postgres: Record<string, unknown>;
    dataset_type?: string | null;
  }) => api.post("/data/connections/", body),
  sync: (dataSourceId: string) =>
    api.post(`/data/connections/${dataSourceId}/sync`),
  tables: (dataSourceId: string) =>
    api.get(`/data/connections/${dataSourceId}/tables`),
  previewSaved: (dataSourceId: string, table: string, rows = 50) =>
    api.get(`/data/connections/${dataSourceId}/preview`, {
      params: { table, rows },
    }),
};

// ── Models ───────────────────────────────────────────────────────────────────

export const modelsApi = {
  train: (datasetId: string, config: Record<string, unknown> = {}) =>
    api.post("/models/train", { dataset_id: datasetId, ...config }),
  listRuns: (params?: Record<string, unknown>) =>
    api.get("/models/runs", { params }),
  getRun: (runId: string) => api.get(`/models/runs/${runId}`),
  getPipeline: (runId: string) => api.get(`/models/runs/${runId}/pipeline`),
  promote: (runId: string) => api.post(`/models/runs/${runId}/promote`),
  listVersions: (activeOnly = false) =>
    api.get("/models/registry", { params: { active_only: activeOnly } }),
  predict: (modelRunId: string, horizon?: number) =>
    api.post("/models/predict", { model_run_id: modelRunId, horizon }),
};

// ── Agents / Chat ────────────────────────────────────────────────────────────

export const agentsApi = {
  sendMessage: (message: string, sessionId?: string, datasetId?: string) =>
    api.post("/agents/chat", { message, session_id: sessionId, dataset_id: datasetId }),
  listSessions: (datasetId?: string) =>
    api.get("/agents/chat/sessions", { params: { dataset_id: datasetId } }),
  getMessages: (sessionId: string) =>
    api.get(`/agents/chat/sessions/${sessionId}/messages`),
  deleteSession: (sessionId: string) =>
    api.delete(`/agents/chat/sessions/${sessionId}`),
  startPipeline: (datasetId: string, config: Record<string, unknown> = {}) =>
    api.post("/agents/pipeline/start", { dataset_id: datasetId, config }),
};

// ── Monitoring ───────────────────────────────────────────────────────────────

export const monitoringApi = {
  summary: () => api.get("/monitoring/summary"),
  drift: (modelVersionId: string, days = 30) =>
    api.get(`/monitoring/drift/${modelVersionId}`, { params: { days } }),
  alerts: (hours = 24) => api.get("/monitoring/alerts", { params: { hours } }),
  createSchedule: (config: Record<string, unknown>) =>
    api.post("/monitoring/schedules", config),
  listSchedules: (activeOnly = false) =>
    api.get("/monitoring/schedules", { params: { active_only: activeOnly } }),
  toggleSchedule: (scheduleId: string) =>
    api.patch(`/monitoring/schedules/${scheduleId}/toggle`),
  deleteSchedule: (scheduleId: string) =>
    api.delete(`/monitoring/schedules/${scheduleId}`),
};

// ── WebSocket Chat ───────────────────────────────────────────────────────────

export function createChatWebSocket(
  onMessage: (data: Record<string, unknown>) => void,
  onClose?: () => void,
): WebSocket {
  const wsUrl = API_BASE.replace("http", "ws") + "/api/agents/chat/ws";
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      onMessage({ type: "raw", content: event.data });
    }
  };

  ws.onclose = () => onClose?.();

  return ws;
}
