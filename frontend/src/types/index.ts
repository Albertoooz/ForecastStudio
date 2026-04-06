// ── Auth ─────────────────────────────────────────────────────────────────────

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: string;
  tenant_id: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  tenant_id: string;
}

// ── Dataset ──────────────────────────────────────────────────────────────────

export interface Dataset {
  id: string;
  name: string;
  rows: number | null;
  columns: number | null;
  datetime_column: string | null;
  target_column: string | null;
  group_columns: string[] | null;
  frequency: string | null;
  schema_columns: string[] | null;
  dataset_type: "training" | "future_variables";
  linked_dataset_id: string | null;
  created_at: string;
}

// ── Model ────────────────────────────────────────────────────────────────────

export interface ModelRun {
  id: string;
  dataset_id: string;
  status: "queued" | "running" | "completed" | "failed";
  model_type: string;
  horizon: number;
  gap: number;
  best_model_name: string | null;
  metrics: Record<string, number> | null;
  duration_seconds: number | null;
  created_at: string;
}

export interface ModelVersion {
  id: string;
  model_run_id: string;
  version: number;
  is_active: boolean;
  promoted_at: string | null;
  created_at: string;
}

export interface PipelineStep {
  step_name: string;
  agent_name: string | null;
  status: string;
  message: string | null;
  duration_seconds: number | null;
}

// ── Chat ─────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  tool_calls?: Record<string, unknown>[];
  metadata?: Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  dataset_id: string | null;
  message_count: number;
  last_message: string | null;
  created_at: string;
}

// ── Monitoring ───────────────────────────────────────────────────────────────

export interface MetricPoint {
  timestamp: string;
  metric_name: string;
  metric_value: number;
  threshold: number | null;
  alert_triggered: boolean;
}

export interface MonitoringSummary {
  total_model_versions: number;
  active_versions: number;
  total_alerts_24h: number;
  total_schedules: number;
  active_schedules: number;
  recent_metrics: MetricPoint[];
}

export interface Schedule {
  id: string;
  model_version_id: string;
  schedule_type: string;
  cron_expression: string;
  is_active: boolean;
  next_run_at: string | null;
  last_run_at: string | null;
}
