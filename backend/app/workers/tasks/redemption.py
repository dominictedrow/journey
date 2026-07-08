"""Module E task: on-chain redemption of settled live positions."""

from app.blockchain.redemption import redeem_settled_positions
from app.database.base import async_session_factory
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async


@celery_app.task(name="copy.redeem_settled_positions")
def redeem_settled_positions_task() -> list[str]:
    """No-ops cheaply on a paper-only deployment (see `redeem_settled_positions`)."""

    async def job() -> list[str]:
        async with async_session_factory() as session:
            return await redeem_settled_positions(session)

    return run_async(job())
