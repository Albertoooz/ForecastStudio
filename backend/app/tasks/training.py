"""
Training tasks — async model training via Celery.
"""

import asyncio
import logging
from uuid import UUID

from sqlalchemy import select

from app.db.models import Dataset, ModelRun, PipelineStep
from app.db.session import async_session_factory
from app.tasks import celery_app

logger = logging.getLogger(__name__)


async def _persist_run_failure(model_run_id: str, error_message: str) -> None:
    """Commit failed status in a fresh session so it survives rollback of the main training transaction."""
    async with async_session_factory() as db:
        try:
            result = await db.execute(select(ModelRun).where(ModelRun.id == UUID(model_run_id)))
            run = result.scalar_one_or_none()
            if not run:
                logger.error("ModelRun %s not found when persisting failure", model_run_id)
                return
            run.status = "failed"
            run.error_message = error_message[:2000]
            db.add(
                PipelineStep(
                    model_run_id=run.id,
                    step_name="error",
                    agent_name=None,
                    status="failed",
                    message=error_message[:500],
                )
            )
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("Could not persist failure state for model_run %s", model_run_id)


async def _run_training_task(model_run_id: str) -> None:
    """
    Load data and run ModelService.run_training without an outer transaction.
    On success, commit. On error, rollback the training transaction and persist failure separately.
    """
    from app.services.model_service import ModelService
    from app.storage.blob import get_blob_storage
    from forecaster.utils.tabular import read_df_from_bytes

    async with async_session_factory() as db:
        try:
            run = (
                await db.execute(select(ModelRun).where(ModelRun.id == UUID(model_run_id)))
            ).scalar_one()

            dataset = (
                await db.execute(select(Dataset).where(Dataset.id == run.dataset_id))
            ).scalar_one()

            blob = get_blob_storage()
            try:
                data_bytes = blob.download_bytes("datasets", dataset.blob_path)
                df = read_df_from_bytes(data_bytes, path_hint=dataset.blob_path or "")
            except Exception as blob_err:
                raise ValueError(
                    f"Cannot load dataset from blob: {dataset.blob_path}"
                ) from blob_err

            await ModelService.run_training(UUID(model_run_id), df, db)
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.exception("Training task failed for run %s", model_run_id)
            await _persist_run_failure(model_run_id, str(e))


async def _run_agent_pipeline_task(model_run_id: str) -> None:
    from app.services.pipeline_service import PipelineService
    from app.storage.blob import get_blob_storage
    from forecaster.utils.tabular import read_df_from_bytes

    async with async_session_factory() as db:
        try:
            run = (
                await db.execute(select(ModelRun).where(ModelRun.id == UUID(model_run_id)))
            ).scalar_one()

            dataset = (
                await db.execute(select(Dataset).where(Dataset.id == run.dataset_id))
            ).scalar_one()

            blob = get_blob_storage()
            data_bytes = blob.download_bytes("datasets", dataset.blob_path)
            df = read_df_from_bytes(data_bytes, path_hint=dataset.blob_path or "")

            await PipelineService.run(UUID(model_run_id), df, db)
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.exception("Agent pipeline task failed for run %s", model_run_id)
            await _persist_run_failure(model_run_id, str(e))


@celery_app.task(bind=True, name="app.tasks.training.run_training")
def run_training(_self, model_run_id: str):
    """
    Celery task to execute model training.

    Runs the async ModelService.run_training in a sync context.
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run_training_task(model_run_id))
    finally:
        loop.close()


@celery_app.task(bind=True, name="app.tasks.training.run_agent_pipeline")
def run_agent_pipeline(_self, model_run_id: str):
    """
    Full agent pipeline task — data analysis, feature engineering,
    model selection, training, evaluation.
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run_agent_pipeline_task(model_run_id))
    finally:
        loop.close()
