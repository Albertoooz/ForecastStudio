"""
Model Service — orchestrate training and prediction using the forecaster library.

Uses the same LangGraph-backed workflow as the Streamlit app, but runs
synchronously inside a Celery task (no Streamlit / no HITL interrupts).
"""

import pickle
import time
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Dataset, Forecast, ModelRun, PipelineStep
from app.storage.blob import get_blob_storage


class ModelService:
    """Stateless service for model training and prediction."""

    # ── Training ──────────────────────────────────────────────────────────────

    @staticmethod
    async def run_training(
        model_run_id: UUID,
        df: pd.DataFrame,
        db: AsyncSession,
    ) -> ModelRun:
        """
        Execute the full training pipeline via forecaster.agents.workflow_engine.

        Steps logged to PipelineStep; results saved to ModelRun and blob storage.
        """
        result = await db.execute(select(ModelRun).where(ModelRun.id == model_run_id))
        run = result.scalar_one()

        dataset_result = await db.execute(select(Dataset).where(Dataset.id == run.dataset_id))
        dataset = dataset_result.scalar_one()

        run.status = "running"
        await db.flush()

        start = time.time()
        config = run.config_json or {}
        datetime_col = config.get("datetime_column") or dataset.datetime_column
        target_col = config.get("target_column") or dataset.target_column
        group_cols = config.get("group_columns") or (
            dataset.group_columns if isinstance(dataset.group_columns, list) else []
        )
        horizon = run.horizon or 12
        gap = run.gap or 0
        model_type = run.model_type or "auto"
        frequency = config.get("frequency") or dataset.frequency

        if not datetime_col or not target_col:
            run.status = "failed"
            run.error_message = "datetime_column or target_column not configured on dataset"
            await db.flush()
            return run

        # ── Progress callback → PipelineStep rows ─────────────────────────────
        step_times: dict[str, float] = {}

        async def _log(step: str, status: str, msg: str = ""):
            nonlocal step_times
            duration = None
            if status == "running":
                step_times[step] = time.time()
            elif step in step_times:
                duration = round(time.time() - step_times[step], 2)

            step_row = PipelineStep(
                model_run_id=run.id,
                step_name=step,
                agent_name=_STEP_AGENTS.get(step),
                status=status,
                message=msg or None,
                duration_seconds=duration,
            )
            db.add(step_row)
            await db.flush()

        # Sync callback for run_forecast_workflow
        import asyncio

        def progress_cb(step: str, status: str, msg: str = ""):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_log(step, status, msg))

        # ── Run workflow ──────────────────────────────────────────────────────
        try:
            from forecaster.agents.workflow_engine import run_forecast_workflow

            wf_result = run_forecast_workflow(
                df=df,
                datetime_column=datetime_col,
                target_column=target_col,
                horizon=horizon,
                gap=gap,
                group_cols=group_cols,
                frequency=frequency,
                model_type=model_type,
                progress_callback=progress_cb,
            )

            if not wf_result.success:
                failed_steps = [k for k, v in wf_result.steps.items() if v.status == "failed"]
                raise RuntimeError(f"Workflow failed at: {', '.join(failed_steps)}")

            # ── Persist metrics ───────────────────────────────────────────────
            fr = wf_result.forecast_result
            metrics: dict = {}
            if fr:
                bm = fr.baseline_metrics or {}
                metrics = {
                    "rmse": round(bm.get("rmse_model", 0), 4),
                    "rmse_naive": round(bm.get("rmse_naive", 0), 4),
                    "mape": round(bm.get("mape_model", 0), 4),
                    "rmse_improvement_pct": round(bm.get("rmse_improvement_pct", 0), 2),
                    "health_score": round(fr.health_score or 0, 1),
                    "n_predictions": len(fr.predictions),
                }

            run.metrics_json = metrics
            run.best_model_name = wf_result.best_model_name

            # ── Persist model artifact ────────────────────────────────────────
            if wf_result.best_model:
                blob = get_blob_storage()
                model_bytes = pickle.dumps(wf_result.best_model)
                model_blob = f"{run.tenant_id}/{run.id}/model.pkl"
                blob.upload_bytes("models", model_blob, model_bytes)
                run.model_blob_path = model_blob

            # ── Persist forecast predictions ──────────────────────────────────
            if fr:
                forecast_row = Forecast(
                    model_run_id=run.id,
                    tenant_id=run.tenant_id,
                    horizon=horizon,
                    predictions_json={
                        "dates": fr.dates,
                        "predictions": fr.predictions,
                        "model_name": fr.model_name,
                        "group_info": fr.group_info,
                        "health_score": fr.health_score,
                        "warnings": fr.warnings,
                    },
                )
                db.add(forecast_row)

            # ── Feature config ────────────────────────────────────────────────
            run.features_json = wf_result.features_config

            run.status = "completed"
            run.duration_seconds = round(time.time() - start, 2)
            await db.flush()

        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.duration_seconds = round(time.time() - start, 2)
            db.add(
                PipelineStep(
                    model_run_id=run.id,
                    step_name="error",
                    status="failed",
                    message=str(e)[:500],
                )
            )
            await db.flush()
            raise

        return run

    # ── Prediction ────────────────────────────────────────────────────────────

    @staticmethod
    async def run_forecast(forecast_id: UUID, db: AsyncSession) -> Forecast:
        """Re-run predictions for an existing trained model."""
        result = await db.execute(select(Forecast).where(Forecast.id == forecast_id))
        forecast = result.scalar_one()

        run_result = await db.execute(select(ModelRun).where(ModelRun.id == forecast.model_run_id))
        run = run_result.scalar_one()

        if not run.model_blob_path:
            forecast.predictions_json = {"error": "No model artifact found"}
            await db.flush()
            return forecast

        blob = get_blob_storage()
        model_bytes = blob.download_bytes("models", run.model_blob_path)
        model = pickle.loads(model_bytes)

        horizon = forecast.horizon or run.horizon or 12
        try:
            from forecaster.models.mlforecast_models import MLForecastModel

            if isinstance(model, MLForecastModel):
                pred_df = model.predict(horizon)
                dates = pred_df.get("ds", pred_df.get("datetime", pd.Series())).astype(str).tolist()
                predictions = pred_df["prediction"].tolist()
            else:
                simple = model.predict(horizon)
                dates = simple.dates
                predictions = simple.predictions

            forecast.predictions_json = {"dates": dates, "predictions": predictions}
        except Exception as e:
            forecast.predictions_json = {"error": str(e)}

        await db.flush()
        return forecast


_STEP_AGENTS = {
    "analysis": "DataAnalyzerAgent",
    "preparation": "ForecastWizard",
    "features": "FeatureEngineerAgent",
    "model_selection": "ModelSelectorAgent",
    "training": "ModelAgent",
    "evaluation": "Evaluator",
    "forecast": "ModelAgent",
}
