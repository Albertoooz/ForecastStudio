"""
Pipeline Service — thin wrapper delegating to ModelService.
"""

from uuid import UUID

import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ModelRun, PipelineStep
from app.services.model_service import ModelService


class PipelineService:
    @staticmethod
    async def run(model_run_id: UUID, df: pl.DataFrame, db: AsyncSession) -> ModelRun:
        return await ModelService.run_training(model_run_id, df, db)

    @staticmethod
    async def get_pipeline_status(model_run_id: UUID, db: AsyncSession) -> dict:
        run_result = await db.execute(select(ModelRun).where(ModelRun.id == model_run_id))
        run = run_result.scalar_one_or_none()
        if not run:
            return {"status": "not_found"}

        steps_result = await db.execute(
            select(PipelineStep)
            .where(PipelineStep.model_run_id == model_run_id)
            .order_by(PipelineStep.created_at)
        )
        steps = steps_result.scalars().all()

        return {
            "model_run_id": str(run.id),
            "status": run.status,
            "model_type": run.model_type,
            "best_model": run.best_model_name,
            "metrics": run.metrics_json,
            "duration": run.duration_seconds,
            "steps": [
                {
                    "name": s.step_name,
                    "agent": s.agent_name,
                    "status": s.status,
                    "message": s.message,
                    "duration": s.duration_seconds,
                }
                for s in steps
            ],
        }
