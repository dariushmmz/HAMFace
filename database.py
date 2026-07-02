"""database.py — SQLite persistence layer for HAMFace tracking sessions.

Schema
------
sessions      — one row per tracking run (webcam / video / image batch)
detections    — one row per face detection event, FK → sessions
persons       — denormalised person lookup (synced from label_map at write time)

Switch to MySQL by replacing the sqlite3 calls with a SQLAlchemy engine;
the schema DDL is intentionally ANSI-compatible.
"""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).parent / "hamface_tracker.db"

# One connection per thread — safe for FastAPI's thread-pool workers and
# asyncio tasks that run sync code via run_in_executor.
_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        _local.conn = con
    return _local.conn


@contextmanager
def _tx():
    """Yield a cursor inside an auto-commit / rollback transaction."""
    con = _conn()
    cur = con.cursor()
    try:
        yield cur
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        cur.close()


# ── Schema init ───────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist. Safe to call at every startup."""
    with _tx() as cur:
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT    NOT NULL,          -- 'webcam' | 'video' | 'image'
                label       TEXT,                      -- filename / cam-id / custom label
                started_at  TEXT    NOT NULL,
                ended_at    TEXT,
                frame_count INTEGER DEFAULT 0,
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                ts          TEXT    NOT NULL,           -- ISO-8601 UTC
                person_name TEXT    NOT NULL,
                score       REAL    NOT NULL,
                bbox_x1     INTEGER,
                bbox_y1     INTEGER,
                bbox_x2     INTEGER,
                bbox_y2     INTEGER,
                frame_no    INTEGER,
                is_unknown  INTEGER NOT NULL DEFAULT 0  -- 1 = "unknown" label
            );

            CREATE INDEX IF NOT EXISTS idx_det_session  ON detections(session_id);
            CREATE INDEX IF NOT EXISTS idx_det_person   ON detections(person_name);
            CREATE INDEX IF NOT EXISTS idx_det_ts       ON detections(ts);
        """)


# ── Session helpers ───────────────────────────────────────────────────────────

def create_session(source: str, label: str = "", notes: str = "") -> int:
    """Insert a new session row and return its id."""
    now = datetime.now(timezone.utc).isoformat()
    with _tx() as cur:
        cur.execute(
            "INSERT INTO sessions (source, label, started_at, notes) VALUES (?,?,?,?)",
            (source, label, now, notes),
        )
        return cur.lastrowid


def end_session(session_id: int, frame_count: int = 0) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _tx() as cur:
        cur.execute(
            "UPDATE sessions SET ended_at=?, frame_count=? WHERE id=?",
            (now, frame_count, session_id),
        )


def get_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    cur = _conn().cursor()
    rows = cur.execute(
        """SELECT s.*,
                  COUNT(d.id)              AS total_detections,
                  SUM(d.is_unknown=0)      AS known_detections,
                  SUM(d.is_unknown=1)      AS unknown_detections
           FROM   sessions s
           LEFT JOIN detections d ON d.session_id = s.id
           GROUP  BY s.id
           ORDER  BY s.started_at DESC
           LIMIT ? OFFSET ?""",
        (limit, offset),
    ).fetchall()
    cur.close()
    return [dict(r) for r in rows]


def get_session(session_id: int) -> Optional[dict]:
    cur = _conn().cursor()
    row = cur.execute(
        "SELECT * FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    cur.close()
    return dict(row) if row else None


def delete_session(session_id: int) -> None:
    with _tx() as cur:
        cur.execute("DELETE FROM sessions WHERE id=?", (session_id,))


# ── Detection helpers ─────────────────────────────────────────────────────────

def log_detection(
    session_id: int,
    person_name: str,
    score: float,
    bbox: Optional[list[int]] = None,
    frame_no: Optional[int] = None,
) -> int:
    """Insert one detection row and return its id."""
    now        = datetime.now(timezone.utc).isoformat()
    is_unknown = 1 if person_name == "unknown" else 0
    bbox       = bbox or [None, None, None, None]
    with _tx() as cur:
        cur.execute(
            """INSERT INTO detections
               (session_id, ts, person_name, score, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                frame_no, is_unknown)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (session_id, now, person_name, score,
             bbox[0], bbox[1], bbox[2], bbox[3], frame_no, is_unknown),
        )
        return cur.lastrowid


def get_detections(
    session_id: int,
    limit: int = 200,
    offset: int = 0,
    person_name: Optional[str] = None,
) -> list[dict]:
    cur    = _conn().cursor()
    params: list = [session_id]
    where  = "session_id=?"
    if person_name:
        where += " AND person_name=?"
        params.append(person_name)
    rows = cur.execute(
        f"SELECT * FROM detections WHERE {where} ORDER BY ts DESC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    ).fetchall()
    cur.close()
    return [dict(r) for r in rows]


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_person_summary(session_id: Optional[int] = None) -> list[dict]:
    """Per-person detection count + avg score, optionally scoped to a session."""
    cur    = _conn().cursor()
    where  = "WHERE session_id=?" if session_id else ""
    params = (session_id,) if session_id else ()
    rows   = cur.execute(
        f"""SELECT person_name,
                   COUNT(*)      AS detections,
                   AVG(score)    AS avg_score,
                   MIN(ts)       AS first_seen,
                   MAX(ts)       AS last_seen
            FROM   detections
            {where}
            GROUP  BY person_name
            ORDER  BY detections DESC""",
        params,
    ).fetchall()
    cur.close()
    return [dict(r) for r in rows]


def get_timeline(session_id: int, bucket_seconds: int = 10) -> list[dict]:
    """
    Return a time-bucketed detection count for charting.

    Buckets *bucket_seconds*-wide; returns list of {bucket, count, known, unknown}.
    """
    cur  = _conn().cursor()
    rows = cur.execute(
        """SELECT CAST(strftime('%s', ts) / ? AS INTEGER) * ? AS bucket,
                  COUNT(*)             AS count,
                  SUM(is_unknown=0)    AS known,
                  SUM(is_unknown=1)    AS unknown
           FROM   detections
           WHERE  session_id=?
           GROUP  BY bucket
           ORDER  BY bucket""",
        (bucket_seconds, bucket_seconds, session_id),
    ).fetchall()
    cur.close()
    return [dict(r) for r in rows]


def get_global_stats() -> dict:
    """Aggregate stats across all sessions."""
    cur = _conn().cursor()
    row = cur.execute(
        """SELECT COUNT(DISTINCT s.id)   AS total_sessions,
                  COUNT(d.id)            AS total_detections,
                  SUM(d.is_unknown=0)    AS known_detections,
                  SUM(d.is_unknown=1)    AS unknown_detections,
                  COUNT(DISTINCT d.person_name) AS unique_persons
           FROM sessions s
           LEFT JOIN detections d ON d.session_id = s.id"""
    ).fetchone()
    cur.close()
    return dict(row) if row else {}
