"""
Professional Chat Agent with OpenAI Function Calling and Streaming.

This is the upgraded version using:
- OpenAI structured outputs / function calling
- Streaming responses
- Thinking blocks (chain of thought)
- Tool registry pattern
"""

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from forecaster.agents.tool_registry import get_tool_registry
from forecaster.core.session import ForecastSession
from forecaster.utils.llm_env import get_openai_compatible_settings
from forecaster.utils.observability import (
    end_trace,
    langfuse_observation,
    log_step,
    start_trace,
)
from forecaster.utils.streamlit_optional import get_session_state

# Load environment
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


class ChatAgentV2:
    """
    Professional Chat Agent with function calling and streaming.

    Features:
    - OpenAI function calling for tool use
    - Streaming responses for real-time feedback
    - Thinking blocks to show reasoning
    - Tool registry integration
    - Error recovery
    """

    def __init__(self):
        api_key, base_url, model = get_openai_compatible_settings()

        if not api_key:
            raise ValueError(
                "LLM API key not found. Set LLM_API_KEY, or legacy DEEPSEEK_API_KEY / OPENAI_API_KEY"
            )

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        self.model = model
        self.tool_registry = get_tool_registry()

    # Tools whose results are returned directly to the LLM (not deferred to app.py)
    IMMEDIATE_TOOLS = frozenset(
        {
            "describe_data",
            "show_dtypes",
            "show_head",
            "value_counts",
            "show_correlation",
            "show_missing_summary",
            "get_model_info",
            "show_pipeline_status",
            "list_datasets",
        }
    )

    MAX_TOOL_ITERATIONS = 5

    def process(
        self, session: ForecastSession, user_message: str, stream: bool = False
    ) -> dict[str, Any]:
        """
        Process user message with a multi-turn tool-call loop.

        Inspect tools (show_head, describe_data, …) are executed immediately
        and their results are fed BACK to the LLM so it can reason over
        actual data before deciding on further actions.

        Action tools (combine_datetime, set_lags, run_forecast, …) are
        collected and returned to app.py for execution.
        """
        start_trace("agent_process", user_message=user_message)

        context = self._build_context(session)
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nUSER:\n{user_message}"},
        ]

        tools = self.tool_registry.get_openai_tools()
        collected_actions: list[dict] = []
        all_tool_calls: list[dict] = []

        try:
            final_text = ""
            thinking = None

            with langfuse_observation(
                as_type="span",
                name="chat.process",
                input={
                    "user_message": user_message,
                    "session_id": getattr(session, "trace_id", None),
                    "model": self.model,
                },
            ) as chat_span:
                for iteration in range(self.MAX_TOOL_ITERATIONS):
                    log_step(
                        "llm_call",
                        "openai_chat",
                        {
                            "model": self.model,
                            "iteration": iteration,
                            "tools_count": len(tools),
                        },
                    )

                    with langfuse_observation(
                        as_type="generation",
                        name="llm.chat.completions",
                        model=self.model,
                        input={
                            "iteration": iteration,
                            "tools_count": len(tools),
                            "tool_names": [t.get("function", {}).get("name") for t in tools or []],
                        },
                    ) as generation:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=tools if tools else None,
                            tool_choice="auto",
                            temperature=0.3,
                        )

                    message = response.choices[0].message
                    if generation is not None:
                        generation.update(
                            output={
                                "has_tool_calls": bool(message.tool_calls),
                                "tool_calls_count": len(message.tool_calls or []),
                                "content_present": bool(message.content),
                            }
                        )

                    # ── No tool calls → final response ──
                    if not message.tool_calls:
                        final_text = message.content or ""
                        break

                    # ── Process tool calls ──
                    # Append assistant turn (with tool_calls) so the next
                    # request includes the full conversation history.
                    messages.append(message)

                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            arguments = json.loads(tool_call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            arguments = {}

                        all_tool_calls.append(
                            {
                                "id": tool_call.id,
                                "name": tool_name,
                                "arguments": arguments,
                            }
                        )
                        log_step("tool_call", tool_name, arguments)

                        if tool_name in self.IMMEDIATE_TOOLS:
                            # Execute inline → LLM sees real data
                            data_str = self._execute_inspect_tool(tool_name, arguments, session)
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": data_str,
                                }
                            )
                        else:
                            # Deferred → queue action descriptor for app.py
                            tool_result = self.tool_registry.execute_tool(tool_name, arguments)
                            if tool_result["success"]:
                                out = tool_result["result"]
                                if isinstance(out, dict):
                                    if "action" in out:
                                        collected_actions.append(out)
                                    elif "operation" in out:
                                        collected_actions.append(
                                            {
                                                "action": "data_operation",
                                                "config": out,
                                            }
                                        )
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": json.dumps(
                                            {
                                                "status": "ok",
                                                "message": f"'{tool_name}' queued — will execute after your response.",
                                            }
                                        ),
                                    }
                                )
                            else:
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": json.dumps(
                                            {
                                                "status": "error",
                                                "error": tool_result.get("error", "unknown"),
                                            }
                                        ),
                                    }
                                )

                    # If the model already generated text alongside deferred
                    # tool calls, treat it as the final response.
                    if collected_actions and (message.content or "").strip():
                        final_text = message.content or ""
                        break
                else:
                    # Exhausted iterations — use whatever the model last said
                    final_text = message.content or "" if message else ""

                # ── Extract <thinking> tags ──
                if "<thinking>" in final_text and "</thinking>" in final_text:
                    t_start = final_text.index("<thinking>") + len("<thinking>")
                    t_end = final_text.index("</thinking>")
                    thinking = final_text[t_start:t_end].strip()
                    final_text = (
                        final_text[: final_text.index("<thinking>")]
                        + final_text[final_text.index("</thinking>") + len("</thinking>") :]
                    ).strip()

                if chat_span is not None:
                    chat_span.update(
                        output={
                            "response": final_text,
                            "tool_calls": len(all_tool_calls),
                            "actions": len(collected_actions),
                        }
                    )

            end_trace(success=True)
            return {
                "response": final_text,
                "thinking": thinking,
                "tool_calls": all_tool_calls,
                "actions": collected_actions,
            }

        except Exception as e:
            import traceback

            print(f"[ChatAgentV2] Error: {e}")
            print(f"[ChatAgentV2] Traceback: {traceback.format_exc()}")
            end_trace(success=False, error=str(e))
            return {
                "response": f"Wystąpił błąd: {e}. Spróbuj ponownie.",
                "thinking": None,
                "tool_calls": all_tool_calls,
                "actions": collected_actions,
            }

    # ------------------------------------------------------------------
    #  Inline execution of inspect tools (data returned to LLM)
    # ------------------------------------------------------------------

    def _execute_inspect_tool(
        self, tool_name: str, arguments: dict, session: ForecastSession
    ) -> str:
        """Run an inspect tool and return a concise text result for the LLM."""
        # Model / pipeline tools don't need a DataFrame
        if tool_name == "list_datasets":
            return self._inspect_list_datasets(session)
        if tool_name == "get_model_info":
            return self._inspect_model_info(session)
        if tool_name == "show_pipeline_status":
            return self._inspect_pipeline_status()

        # All other inspect tools need the DataFrame
        df = getattr(session, "current_df", None)
        if df is None:
            df = get_session_state().get("current_dataframe")
        if df is None:
            return "No data loaded."

        try:
            if tool_name == "show_head":
                n = min(arguments.get("n", 10), 30)
                subset = df.head(n)
                extra = ""
                if len(df.columns) > 15:
                    subset = subset.iloc[:, :15]
                    extra = f"\n({len(df.columns) - 15} more columns not shown)"
                return (
                    f"{subset.to_string()}\n"
                    f"\nShape: {len(df):,} rows × {len(df.columns)} cols{extra}"
                )

            elif tool_name == "describe_data":
                cols = arguments.get("columns")
                target = (
                    df[cols]
                    if cols and all(c in df.columns for c in cols)
                    else df.select_dtypes(include="number")
                )
                if len(target.columns) > 12:
                    target = target.iloc[:, :12]
                return target.describe().to_string()

            elif tool_name == "show_dtypes":
                lines = []
                for col in df.columns:
                    nulls = int(df[col].isnull().sum())
                    null_info = (
                        f"  ({nulls:,} nulls, {nulls / len(df) * 100:.1f}%)" if nulls else ""
                    )
                    lines.append(f"  {col}: {df[col].dtype}{null_info}")
                return f"Columns ({len(df.columns)}):\n" + "\n".join(lines)

            elif tool_name == "value_counts":
                column = arguments.get("column", "")
                top_n = arguments.get("top_n", 20)
                if column not in df.columns:
                    return f"Column '{column}' not found. Available: {', '.join(df.columns[:20])}"
                vc = df[column].value_counts().head(top_n)
                return f"Value counts for '{column}':\n{vc.to_string()}"

            elif tool_name == "show_correlation":
                numeric = df.select_dtypes(include="number")
                target = session.data_info.target_column if session.data_info else None
                cols = arguments.get("columns")
                if cols:
                    valid = [c for c in cols if c in numeric.columns]
                    if valid:
                        numeric = numeric[valid]
                if target and target in numeric.columns:
                    corr = (
                        numeric.corr()[target]
                        .drop(target, errors="ignore")
                        .abs()
                        .sort_values(ascending=False)
                        .head(10)
                    )
                    return f"Correlation with '{target}':\n{corr.to_string()}"
                return f"Correlation matrix:\n{numeric.corr().to_string()}"

            elif tool_name == "show_missing_summary":
                missing = df.isnull().sum()
                missing = missing[missing > 0].sort_values(ascending=False)
                if missing.empty:
                    return "No missing values in the dataset."
                total = int(missing.sum())
                pct = total / (len(df) * len(df.columns)) * 100
                return (
                    f"Missing values:\n{missing.to_string()}\n\nTotal: {total:,} cells ({pct:.1f}%)"
                )

        except Exception as exc:
            return f"Error executing {tool_name}: {exc}"

        return f"Unknown inspect tool: {tool_name}"

    def _inspect_list_datasets(self, session: ForecastSession) -> str:
        """Format tenant dataset catalog for the LLM (from API-injected available_datasets)."""
        cats = session.available_datasets or []
        if not cats:
            return "No dataset catalog loaded (empty workspace or API did not attach available_datasets)."
        lines: list[str] = [f"Total datasets: {len(cats)}\n"]
        for d in cats:
            did = d.get("id", "?")
            name = d.get("name", "?")
            rows = d.get("rows")
            cols = d.get("columns")
            st = d.get("source_type") or "unknown"
            sync = d.get("sync_status") or ""
            dstype = d.get("dataset_type") or "training"
            dt = d.get("datetime_column") or "—"
            tgt = d.get("target_column") or "—"
            qot = d.get("query_or_table")
            qline = (
                f"\n    query/table: {qot[:120]}…"
                if qot and len(str(qot)) > 120
                else (f"\n    query/table: {qot}" if qot else "")
            )
            lines.append(
                f"• id={did}\n  name={name!r}\n  type={dstype}  source={st}"
                + (f"  sync={sync}" if sync else "")
                + f"\n  rows={rows}  cols={cols}\n  datetime={dt}  target={tgt}{qline}"
            )
        return "\n".join(lines)

    def _inspect_model_info(self, session: ForecastSession) -> str:
        """Return model information for the LLM."""
        parts: list[str] = []
        try:
            wr = get_session_state().get("agent_workflow_result")

            if wr and getattr(wr, "best_model_name", None):
                parts.append(f"Best model: {wr.best_model_name}")
                for name, res in wr.all_model_results.items():
                    rmse = res.get("holdout_rmse")
                    mape = res.get("holdout_mape")
                    r_s = (
                        f"RMSE={rmse:.2f}"
                        if isinstance(rmse, (int, float)) and rmse < 1e10
                        else "RMSE=N/A"
                    )
                    m_s = (
                        f", MAPE={mape:.1f}%"
                        if isinstance(mape, (int, float)) and mape < 1e6
                        else ""
                    )
                    parts.append(f"  {name}: {r_s}{m_s}")
                if wr.features_config:
                    fc = wr.features_config
                    parts.append(
                        f"Features: lags={fc.get('lags')}, "
                        f"rolling={fc.get('rolling_windows')}, "
                        f"date={fc.get('date_features')}"
                    )
            elif session.forecast_config:
                parts.append(
                    f"Configured: {session.forecast_config.model_type}, horizon={session.forecast_config.horizon}"
                )
                parts.append("Model not yet trained. Run forecast first.")
            else:
                parts.append("No model configured. Set datetime, target, and run forecast.")
        except Exception as exc:
            parts.append(f"Error: {exc}")

        return "\n".join(parts) if parts else "No model information available."

    def _inspect_pipeline_status(self) -> str:
        """Return pipeline status for the LLM."""
        try:
            wr = get_session_state().get("agent_workflow_result")

            if not wr:
                return "No pipeline run yet. Use run_forecast to start."
            parts = [f"Pipeline (total: {wr.total_duration:.1f}s, success: {wr.success}):"]
            for step_name, step in wr.steps.items():
                icon = {"done": "✅", "failed": "❌", "running": "⏳"}.get(step.status, "⬜")
                parts.append(f"  {icon} {step_name}: {step.message} ({step.duration:.1f}s)")
            return "\n".join(parts)
        except Exception:
            return "Pipeline status unavailable."

    def _build_context(self, session: ForecastSession) -> str:
        """Build context string for LLM."""
        parts = []

        # Data info - MOST IMPORTANT
        if session.data_info:
            di = session.data_info
            parts.append("=" * 60)
            parts.append("📊 CURRENT DATA STATE")
            parts.append("=" * 60)
            parts.append(f"File: {di.filename}")
            parts.append(f"Shape: {di.n_rows} rows × {len(di.columns)} columns")
            parts.append(f"\nColumns: {', '.join(c.name for c in di.columns)}")

            # Column types
            date_cols = [c.name for c in di.columns if c.dtype in ["datetime64", "datetime"]]
            numeric_cols = [
                c.name for c in di.columns if c.dtype in ["int64", "float64", "int32", "float32"]
            ]

            if date_cols:
                parts.append(f"\nDate/Time columns: {', '.join(date_cols)}")
            if numeric_cols:
                parts.append(
                    f"Numeric columns: {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}"
                )

            # Configuration status
            parts.append(
                f"\n{'✅' if di.datetime_column else '❌'} Datetime column: {di.datetime_column or 'NOT SET'}"
            )
            parts.append(
                f"{'✅' if di.target_column else '❌'} Target column: {di.target_column or 'NOT SET'}"
            )

            # Grouping - CRITICAL
            group_cols = di.group_by_columns or []
            if group_cols:
                parts.append(f"📦 GROUPING: {', '.join(group_cols)}")
                parts.append(f"   ⚠️  Feature engineering MUST use .groupby({group_cols}) !")
            else:
                parts.append("📦 Grouping: None (single time series)")

            if di.frequency:
                parts.append(f"⏱️  Frequency: {di.frequency}")
                parts.append(
                    "   ⚠️  CRITICAL: When creating lag features, use shift(n) where n matches frequency!"
                )
                parts.append(
                    "   Example: For 15min data, lag_1_hour = shift(4), lag_1_day = shift(96)"
                )

            if di.date_range:
                parts.append(f"📅 Date range: {di.date_range[0]} to {di.date_range[1]}")

            # Issues
            if di.issues:
                parts.append("\n⚠️  DATA ISSUES:")
                for issue in di.issues[:5]:
                    parts.append(f"   - {issue}")

            # External source metadata (PostgreSQL snapshot, file upload, …)
            if di.dataset_id or di.source_type:
                parts.append("\n" + "=" * 60)
                parts.append("📌 DATA SOURCE INFO (active dataset)")
                parts.append("=" * 60)
                if di.dataset_id:
                    parts.append(f"Dataset id: {di.dataset_id}")
                if di.source_type:
                    parts.append(
                        f"Source: {di.source_type} "
                        f"(postgres/sql = database snapshot; file = CSV/Excel/Parquet upload)"
                    )
                if di.sync_status:
                    parts.append(f"Sync status: {di.sync_status}")
                if di.last_sync_at:
                    parts.append(f"Last sync: {di.last_sync_at}")
                if di.query_or_table:
                    q = str(di.query_or_table)
                    parts.append(f"Query / table: {q[:300]}{'…' if len(q) > 300 else ''}")
        else:
            parts.append("❌ NO DATA LOADED")

        # All datasets in workspace (injected by chat API)
        ads = session.available_datasets
        if ads:
            parts.append("\n" + "=" * 60)
            parts.append("📚 AVAILABLE DATASETS (switch with switch_dataset)")
            parts.append("=" * 60)
            for d in ads[:30]:
                marker = (
                    " ← ACTIVE"
                    if session.active_dataset_id
                    and str(d.get("id")) == str(session.active_dataset_id)
                    else ""
                )
                parts.append(
                    f"  • {d.get('name')}  id={d.get('id')}  "
                    f"src={d.get('source_type') or '?'}  rows={d.get('rows')}{marker}"
                )
            if len(ads) > 30:
                parts.append(f"  … and {len(ads) - 30} more (use list_datasets for full detail)")

        # Model config
        if session.forecast_config:
            mc = session.forecast_config
            parts.append(f"\n{'=' * 60}")
            parts.append("⚙️  FORECAST CONFIG")
            parts.append("=" * 60)
            parts.append(f"Model: {mc.model_type}")
            parts.append(f"Horizon: {mc.horizon}")

        ss = get_session_state()
        override_parts = []
        for key, label in [
            ("feature_override_lags", "Lag override"),
            ("feature_override_rolling", "Rolling window override"),
            ("feature_override_date", "Date feature override"),
            ("feature_override_ewm", "EWM override"),
            ("model_override_hyperparams", "Model hyperparameter override"),
        ]:
            val = ss.get(key)
            if val is not None:
                override_parts.append(f"  {label}: {val}")
        if override_parts:
            parts.append("\n🔧 ACTIVE OVERRIDES (will apply on next forecast):")
            parts.extend(override_parts)

        # Agent context (from new multi-agent pipeline)
        try:
            ctx = ss.get("agent_context")
            if ctx is not None:
                parts.append(f"\n{'=' * 60}")
                parts.append("🤖 AGENT PIPELINE RESULTS")
                parts.append("=" * 60)

                # Feature specifications
                if ctx.feature_specs:
                    lag_specs = [s for s in ctx.feature_specs if s.feature_type == "lag"]
                    rolling_specs = [s for s in ctx.feature_specs if "rolling" in s.feature_type]
                    date_specs = [s for s in ctx.feature_specs if s.feature_type == "date_part"]

                    if lag_specs:
                        lags = sorted([s.parameters.get("lag", 0) for s in lag_specs])
                        parts.append(f"🔢 Lag features: {lags}")
                        parts.append("   (FeatureEngineerAgent automatically selected these)")

                    if rolling_specs:
                        windows = sorted({s.parameters.get("window", 0) for s in rolling_specs})
                        parts.append(f"📊 Rolling windows: {windows}")

                    if date_specs:
                        date_names = [s.parameters.get("part", s.name) for s in date_specs]
                        parts.append(f"📅 Date features: {', '.join(date_names)}")

                    parts.append(
                        "\n   ℹ️  These features were AUTO-GENERATED by FeatureEngineerAgent"
                    )
                    parts.append("   ℹ️  To change them, re-run the forecast (agent will adapt)")

                # Model recommendation
                if ctx.model_spec:
                    parts.append("\n🧠 Model recommendation (ModelSelectorAgent):")
                    parts.append(f"   Recommended: {ctx.model_spec.model_type}")
                    parts.append(f"   Confidence: {ctx.model_spec.confidence * 100:.0f}%")
                    if ctx.model_spec.reasoning:
                        parts.append(f"   Reasoning: {ctx.model_spec.reasoning}")

                # Recent decisions
                if ctx.decision_log:
                    recent = ctx.decision_log[-3:]  # Last 3 decisions
                    parts.append("\n📝 Recent agent decisions:")
                    for d in recent:
                        parts.append(
                            f"   [{d.agent_name}] {d.action} (conf: {d.confidence * 100:.0f}%)"
                        )

        except Exception:
            pass  # No agent context or session state unavailable

        return "\n".join(parts)

    def _get_system_prompt(self) -> str:
        """System prompt — action-first, concise."""
        return """You are a forecasting assistant with DIRECT tool access.

CORE PRINCIPLE — ACT FIRST:
• When the user asks to DO something ("create", "combine", "set", "fill", "run") → call the tool IMMEDIATELY. Do NOT inspect data first unless you genuinely need it to decide.
• When the user asks to KNOW something ("show me", "what does", "how many") → call the appropriate inspect tool. You will see the actual data and can answer.
• You can chain tools: inspect → decide → act, all within one turn.

TOOLS:
  DATA: combine_datetime, create_column, drop_column, rename_column, filter_rows, filter_date_range, sort_data, resample_data
  CONFIG: set_datetime_column, set_target_column, set_grouping_columns, set_horizon, set_gap
  MISSING: fill_missing(column, method), drop_missing_rows, show_missing_summary
  FEATURES: set_lags, set_rolling_windows, set_date_features, set_ewm  (override auto-agents; pass [] to reset)
  MODEL: set_model(model_type, hyperparameters), get_model_info
  PIPELINE: run_forecast, rerun_pipeline_step, show_pipeline_status
  INSPECT: describe_data, show_dtypes, show_head, value_counts, show_correlation
  DATASETS: list_datasets, switch_dataset, resync_dataset

INSPECT vs ACTION tools:
- Inspect tools (show_head, describe_data, show_dtypes, value_counts, show_correlation, show_missing_summary, get_model_info, show_pipeline_status) → you will see their output and can use it in your answer.
- Action tools (combine_datetime, set_lags, run_forecast, etc.) → executed after your response. Say what you did briefly.

RULES:
1. "combine business_date and time_15min_slot" → call combine_datetime(date_column="business_date", time_column="time_15min_slot") RIGHT AWAY.
2. "set lags to 1, 4, 96" → call set_lags(lags=[1, 4, 96]) RIGHT AWAY.
3. "what does the data look like?" → call show_head(), then summarize what you see.
4. Frequency: 15min → 1h=shift(4), 1d=shift(96). Hourly → 1d=shift(24). Daily → 1w=shift(7).
5. Grouping: if group_by_columns set, create_column must use df.groupby(...)['col'].transform(...).
6. Be concise. No walls of text. State what you did and the result.
7. Feature/model overrides take effect on the NEXT run_forecast. Pass [] to clear.
8. Context lists ALL available datasets (PostgreSQL snapshots and file uploads). If the user asks what data exists, use list_datasets for details or read CONTEXT. To analyze another dataset, call switch_dataset(dataset_id) with the UUID from the list; then use inspect tools on the new snapshot.
9. resync_dataset refreshes PostgreSQL-backed snapshots from the database; it does not apply to file uploads.
"""


# Singleton instance
_agent_v2 = None


def get_chat_agent_v2() -> ChatAgentV2:
    """Get the global chat agent instance."""
    global _agent_v2
    if _agent_v2 is None:
        _agent_v2 = ChatAgentV2()
    return _agent_v2
