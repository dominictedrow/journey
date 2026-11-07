from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

_pool = pool.ThreadedConnectionPool(2, 10, settings.DATABASE_URL)


@contextmanager
def get_conn():
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        # A connection dropped by the server (e.g. after macOS sleep) is already
        # closed; rolling it back would raise again. Guard, then discard it below.
        if not conn.closed:
            conn.rollback()
        raise
    finally:
        # Never return a dead connection to the pool — otherwise the next caller
        # gets the same broken handle and every query fails until restart.
        _pool.putconn(conn, close=bool(conn.closed))


def execute(sql, params=()):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def execute_returning(sql, params=()):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def fetchone_dict(sql, params=()):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None


def fetchall_dict(sql, params=()):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
