"""Module D task: the beat-scheduled activity governor."""

from app.autonomy.activity_governor import run_activity_governor
from app.database.base import async_session_factory
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async


@celery_app.task(name="autonomy.activity_governor")
def activity_governor_task() -> dict[str, int | None]:
    """Widen the wallet funnel when trailing 24h trade activity has gone quiet."""

    async def job() -> dict[str, int | None]:
        async with async_session_factory() as session:
            return await run_activity_governor(session)

    return run_async(job())
