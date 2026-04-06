"""
Agents API — Chat (multi-turn LLM + tools), pipeline control.

Provides both REST and WebSocket endpoints for chat interaction.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db.models import ChatSession, User
from app.db.session import get_db

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str  # user | assistant | system | tool
    content: str
    tool_calls: list[dict] | None = None
    metadata: dict | None = None


class ChatRequest(BaseModel):
    session_id: UUID | None = None
    dataset_id: UUID | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: UUID
    reply: str
    actions: list[dict] = []  # proposed actions for the UI to execute/confirm
    tool_calls_made: list[dict] = []
    executed_operations: list[str] = []


class ChatSessionResponse(BaseModel):
    id: UUID
    dataset_id: UUID | None
    message_count: int
    last_message: str | None
    created_at: str


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat_message(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Send a message to the AI chat agent.

    The agent uses the same multi-turn tool-call loop as ChatAgentV2,
    but decoupled from Streamlit.
    """
    # Get or create session
    if body.session_id:
        session = await _get_session(body.session_id, user.tenant_id, db)
    else:
        session = ChatSession(
            tenant_id=user.tenant_id,
            user_id=user.id,
            dataset_id=body.dataset_id,
            messages_json=[],
        )
        db.add(session)
        await db.flush()

    # Append user message
    messages: list[dict] = session.messages_json or []
    messages.append({"role": "user", "content": body.message})

    executed_ops: list[str] = []
    reply = ""
    actions: list[dict] = []
    tool_calls_made: list[dict] = []

    # Build a ForecastSession for the chat agent
    try:
        import sys
        from forecaster.core.session import ForecastSession
        from forecaster.agents.chat_v2 import get_chat_agent_v2

        chat_session = ForecastSession(session_id=str(session.id))

        # Inject dataset context if available
        dataset_id_to_use = session.dataset_id or body.dataset_id
        if dataset_id_to_use:
            from sqlalchemy import select as sa_select
            from app.db.models import Dataset as DatasetModel

            ds_result = await db.execute(
                sa_select(DatasetModel).where(DatasetModel.id == dataset_id_to_use)
            )
            ds = ds_result.scalar_one_or_none()
            if ds:
                try:
                    from app.storage.blob import get_blob_storage
                    from forecaster.core.session import DataInfo, ColumnInfo
                    from pathlib import Path as _Path

                    blob = get_blob_storage()
                    df = blob.download_df("datasets", ds.blob_path)

                    col_infos = [
                        ColumnInfo(name=col, dtype=str(dt)) for col, dt in df.dtypes.items()
                    ]
                    data_info = DataInfo(
                        filepath=_Path(ds.blob_path or ds.name),
                        filename=ds.name,
                        columns=col_infos,
                        datetime_column=ds.datetime_column,
                        target_column=ds.target_column,
                        frequency=ds.frequency,
                        n_rows=ds.row_count or len(df),
                    )
                    chat_session.data_info = data_info
                    # Store df in session so the agent can read it
                    chat_session.current_df = df
                except Exception as load_err:
                    print(f"[agents/chat] Could not load dataset: {load_err}", file=sys.stderr)

        # Restore prior message history (exclude the user message we just appended)
        for m in messages[:-1]:
            role = m.get("role", "")
            if role in ("user", "assistant", "system"):
                chat_session.add_message(role, m.get("content", ""))

        agent = get_chat_agent_v2()
        result = agent.process(chat_session, body.message)

        reply = result.get("response", "")
        actions = result.get("actions", [])
        tool_calls_made = result.get("tool_calls", [])

        # ── Execute data operations returned by the agent ──────────────────
        if actions and dataset_id_to_use:
            for act in actions:
                op_cfg = None
                if act.get("action") == "data_operation":
                    op_cfg = act.get("config", {})
                elif "operation" in act:
                    op_cfg = act

                if op_cfg:
                    op_name = op_cfg.get("operation") or op_cfg.get("action")
                    params = op_cfg.get("parameters", op_cfg.get("params", {}))
                    if op_name:
                        try:
                            executed = await _apply_data_op(
                                dataset_id=UUID(str(dataset_id_to_use)),
                                tenant_id=user.tenant_id,
                                operation=op_name,
                                params=params,
                                also_future_vars=True,
                            )
                            if executed:
                                executed_ops.append(op_name)
                        except Exception as op_err:
                            print(f"[agents/chat] Op {op_name} failed: {op_err}", file=sys.stderr)

            if executed_ops:
                reply += f"\n\n✅ Applied to dataset: {', '.join(executed_ops)}."

    except Exception as e:
        import sys
        import traceback

        traceback.print_exc(file=sys.stderr)
        reply = f"I encountered an error processing your request: {e}"
        tool_calls_made = []
        actions = []
    finally:
        try:
            from forecaster.utils.observability import flush_langfuse

            flush_langfuse()
        except Exception:
            pass

    # Append assistant reply
    messages.append({"role": "assistant", "content": reply})
    session.messages_json = messages
    await db.flush()

    return ChatResponse(
        session_id=session.id,
        reply=reply,
        actions=actions,
        tool_calls_made=tool_calls_made,
        executed_operations=executed_ops,
    )


# ── Data operation executor ───────────────────────────────────────────────────


async def _apply_data_op(
    dataset_id: UUID,
    tenant_id: UUID,
    operation: str,
    params: dict,
    also_future_vars: bool = True,
) -> bool:
    """
    Download dataset, apply a DataOperationsV2 operation, upload back.
    Uses its own DB session so it cannot contaminate the request session.
    """
    from sqlalchemy import select as _select
    from app.db.models import Dataset as DatasetModel
    from app.storage.blob import get_blob_storage
    from app.db.session import async_session_factory
    from forecaster.agents.data_operations_v2 import DataOperations

    blob = get_blob_storage()
    ops = DataOperations()

    def _input_columns_for_operation(op_name: str, op_params: dict) -> list[str]:
        """Return only columns that must already exist for this operation."""
        if op_name == "combine_datetime":
            cols = [op_params.get("date_column"), op_params.get("time_column")]
        elif op_name == "add_column":
            cols = []
        elif op_name == "normalize":
            cols = [op_params.get("column")]
        elif op_name == "aggregate":
            cols = [op_params.get("group_by"), op_params.get("agg_column")]
        elif op_name == "resample":
            cols = [op_params.get("datetime_column"), op_params.get("value_column")]
        elif op_name in {"drop_column", "rename_column", "filter", "sort"}:
            cols = [op_params.get("column"), op_params.get("old_name")]
        elif op_name == "fill_missing":
            cols = [op_params.get("column")]
        elif op_name == "drop_missing":
            cols = [op_params.get("column")]
        elif op_name == "filter_date_range":
            cols = [op_params.get("column")]
        else:
            cols = []
        return [c for c in cols if isinstance(c, str) and c]

    async with async_session_factory() as own_db:
        try:

            async def _process_one(ds: DatasetModel) -> bool:
                try:
                    df = blob.download_df("datasets", ds.blob_path)
                    result = ops.execute_operation(df, operation, params)
                    if not result["success"]:
                        print(
                            f"[chat/op] {operation} failed on {ds.name}: {result['message']}",
                            flush=True,
                        )
                        return False
                    new_df = result["dataframe"]
                    blob.upload_df("datasets", ds.blob_path, new_df)
                    schema = dict(ds.schema_json or {})
                    for col, dt in new_df.dtypes.items():
                        if not str(col).startswith("__"):
                            schema[str(col)] = str(dt)
                    ds.schema_json = schema
                    ds.row_count = len(new_df)
                    ds.column_count = len(new_df.columns)
                    await own_db.flush()
                    return True
                except Exception as e:
                    print(f"[chat/op] error on {ds.name}: {e}", flush=True)
                    return False

            res = await own_db.execute(
                _select(DatasetModel).where(
                    DatasetModel.id == dataset_id,
                    DatasetModel.tenant_id == tenant_id,
                )
            )
            ds = res.scalar_one_or_none()
            if not ds:
                return False

            ok = await _process_one(ds)

            if also_future_vars and ok:
                fv_res = await own_db.execute(
                    _select(DatasetModel).where(DatasetModel.tenant_id == tenant_id)
                )
                required_input_cols = set(_input_columns_for_operation(operation, params))
                for fv_ds in fv_res.scalars().all():
                    schema = fv_ds.schema_json or {}
                    if schema.get("__linked_dataset_id__") == str(dataset_id):
                        try:
                            df_check = blob.download_df("datasets", fv_ds.blob_path)
                            if all(c in df_check.columns for c in required_input_cols):
                                await _process_one(fv_ds)
                        except Exception:
                            pass

            await own_db.commit()
            return ok

        except Exception as e:
            await own_db.rollback()
            print(f"[chat/op] session error: {e}", flush=True)
            return False


@router.get("/chat/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    dataset_id: UUID | None = Query(None),
):
    """List chat sessions for the current user."""
    q = select(ChatSession).where(
        ChatSession.tenant_id == user.tenant_id,
        ChatSession.user_id == user.id,
    )
    if dataset_id:
        q = q.where(ChatSession.dataset_id == dataset_id)
    q = q.order_by(ChatSession.created_at.desc())

    result = await db.execute(q)
    sessions = result.scalars().all()

    return [
        ChatSessionResponse(
            id=s.id,
            dataset_id=s.dataset_id,
            message_count=len(s.messages_json or []),
            last_message=_last_msg(s.messages_json),
            created_at=str(s.created_at),
        )
        for s in sessions
    ]


@router.get("/chat/sessions/{session_id}/messages", response_model=list[ChatMessage])
async def get_messages(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get all messages in a chat session."""
    session = await _get_session(session_id, user.tenant_id, db)
    return [ChatMessage(**m) for m in (session.messages_json or [])]


@router.delete("/chat/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete a chat session."""
    session = await _get_session(session_id, user.tenant_id, db)
    await db.delete(session)


# ── WebSocket (streaming chat) ──────────────────────────────────────────────


@router.websocket("/chat/ws")
async def chat_websocket(websocket: WebSocket):
    """
    WebSocket for streaming chat responses and training progress.

    Protocol:
      Client → { "type": "message", "session_id": "...", "content": "..." }
      Server → { "type": "token", "content": "..." }
      Server → { "type": "tool_call", "name": "...", "args": {...} }
      Server → { "type": "tool_result", "name": "...", "result": "..." }
      Server → { "type": "done", "session_id": "..." }
      Server → { "type": "error", "detail": "..." }
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "message":
                # TODO: authenticate via token in initial handshake
                # TODO: run agent loop with streaming
                await websocket.send_json(
                    {
                        "type": "token",
                        "content": f"[Placeholder] Echo: {data.get('content', '')}",
                    }
                )
                await websocket.send_json({"type": "done"})
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass


# ── Pipeline control ────────────────────────────────────────────────────────


class PipelineStartRequest(BaseModel):
    dataset_id: UUID
    config: dict = {}  # overrides for feature engineering, model selection, etc.


class PipelineStatusResponse(BaseModel):
    model_run_id: UUID
    status: str
    steps: list[dict]


@router.post("/pipeline/start", response_model=PipelineStatusResponse, status_code=202)
async def start_pipeline(
    body: PipelineStartRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Start the full agent pipeline (data analysis → feature engineering →
    model selection → training → evaluation).
    """
    from app.db.models import ModelRun

    model_run = ModelRun(
        tenant_id=user.tenant_id,
        dataset_id=body.dataset_id,
        status="queued",
        model_type="auto",
        config_json=body.config,
    )
    db.add(model_run)
    await db.flush()

    # TODO: dispatch Celery pipeline task
    # from app.tasks.training import run_agent_pipeline
    # run_agent_pipeline.delay(str(model_run.id))

    return PipelineStatusResponse(
        model_run_id=model_run.id,
        status="queued",
        steps=[],
    )


# ── Internal ─────────────────────────────────────────────────────────────────


async def _get_session(session_id: UUID, tenant_id: UUID, db: AsyncSession) -> ChatSession:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.tenant_id == tenant_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Chat session not found")
    return session


def _last_msg(messages: list[dict] | None) -> str | None:
    if not messages:
        return None
    for m in reversed(messages):
        if m.get("role") in ("user", "assistant"):
            content = m.get("content", "")
            return content[:100] if content else None
    return None
