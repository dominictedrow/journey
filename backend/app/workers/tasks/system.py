"""System tasks: smoke-test task proving the worker/broker wiring."""

from app.workers.celery_app import celery_app


@celery_app.task(name="system.ping")
def ping() -> str:
    return "pong"
