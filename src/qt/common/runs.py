"""``run_id`` (D-015) and the ``runs`` table lifecycle."""

from __future__ import annotations

import logging
import re
import secrets
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg

from qt.common.config import Settings
from qt.common.logging import run_id_var, stage_var
from qt.common.paths import REPO_ROOT

_STAGE = re.compile(r"^[a-z][a-z0-9_]{1,20}$")
RUN_ID_RE = re.compile(r"^\d{8}T\d{4}Z-[a-z][a-z0-9_]{1,20}-[0-9a-f]{4}$")

log = logging.getLogger("qt.runs")


def new_run_id(stage: str, now: datetime | None = None) -> str:
    """``YYYYMMDDTHHMMZ-<stage>-<4hex>`` in UTC, e.g. ``20261125T2130Z-collect-a3f9``."""
    if not _STAGE.match(stage):
        raise ValueError(f"bad stage name {stage!r}: want ^[a-z][a-z0-9_]{{1,20}}$")
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return f"{now.astimezone(UTC):%Y%m%dT%H%MZ}-{stage}-{secrets.token_hex(2)}"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class Run:
    run_id: str
    stage: str
    settings: Settings


@contextmanager
def run_context(
    stage: str,
    settings: Settings,
    connect: Callable[[], psycopg.Connection],
) -> Iterator[Run]:
    """Record a run in ``runs`` and tag every log line with its run_id/stage.

    On an exception: the row is marked ``failed`` with the error, the failure is logged at ERROR
    (which pushes a critical alert via the root handler, D-016), and the exception is re-raised.
    """
    run = Run(new_run_id(stage), stage, settings)
    t_run, t_stage = run_id_var.set(run.run_id), stage_var.set(stage)
    try:
        try:
            with connect() as conn:
                conn.execute(
                    "INSERT INTO runs (run_id, stage, profile, config_hash, git_sha) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (run.run_id, stage, settings.profile, settings.config_hash, git_sha()),
                )
        except Exception:
            log.exception("run_start_failed: could not record run in DB")
            raise
        log.info("run_started")
        try:
            yield run
        except BaseException as e:
            err = f"{type(e).__name__}: {e}"
            log.error("run_failed: %s", err, exc_info=True)
            _finish(connect, run.run_id, "failed", err)
            raise
        _finish(connect, run.run_id, "ok", None)
        log.info("run_finished")
    finally:
        run_id_var.reset(t_run)
        stage_var.reset(t_stage)


def _finish(
    connect: Callable[[], psycopg.Connection], run_id: str, status: str, error: str | None
) -> None:
    try:
        with connect() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = now(), status = %s, error = %s WHERE run_id = %s",
                (status, error, run_id),
            )
    except Exception:
        log.exception("run_finish_failed: could not update run %s", run_id)
