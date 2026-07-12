"""Database layer for ADE-Agri.

One DATABASE_URL drives everything:
  local dev  -> sqlite:///ade_agri.db
  production -> postgresql+psycopg://...  (Supabase)
"""
import os
import pathlib
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_URL = f"sqlite:///{ROOT / 'ade_agri.db'}"


def get_engine(url: str | None = None):
    url = url or os.environ.get("DATABASE_URL", DEFAULT_URL)
    return create_engine(url, future=True)


def is_postgres(engine) -> bool:
    return engine.dialect.name == "postgresql"


def init_db(engine) -> None:
    """Create tables if absent. Idempotent."""
    ddl = (ROOT / "schema.sql").read_text(encoding="utf-8")
    if is_postgres(engine):
        # SQLite spelling -> Postgres spelling
        ddl = ddl.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    with engine.begin() as conn:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(stmt))


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def upsert_series(engine, meta: dict) -> None:
    """Register or refresh a series' metadata (feeds the /sources page)."""
    pg = is_postgres(engine)
    conflict = (
        "ON CONFLICT (series_id) DO UPDATE SET "
        "label_vi=EXCLUDED.label_vi, unit=EXCLUDED.unit, frequency=EXCLUDED.frequency, "
        "source_name=EXCLUDED.source_name, source_url=EXCLUDED.source_url, "
        "license_note=EXCLUDED.license_note, is_active=EXCLUDED.is_active"
    ) if pg else "ON CONFLICT(series_id) DO UPDATE SET " \
                 "label_vi=excluded.label_vi, unit=excluded.unit, frequency=excluded.frequency, " \
                 "source_name=excluded.source_name, source_url=excluded.source_url, " \
                 "license_note=excluded.license_note, is_active=excluded.is_active"

    sql = text(
        "INSERT INTO series (series_id, label_vi, unit, frequency, source_name, "
        "source_url, license_note, is_active) VALUES "
        "(:series_id, :label_vi, :unit, :frequency, :source_name, :source_url, "
        ":license_note, :is_active) " + conflict
    )
    with engine.begin() as conn:
        conn.execute(sql, meta)


def upsert_prices(engine, series_id: str, rows: list[tuple]) -> tuple[int, int]:
    """rows = [(obs_date, value), ...]. Returns (added, updated).

    Idempotent by construction: PRIMARY KEY (series_id, obs_date).
    """
    if not rows:
        return 0, 0
    now = utcnow()

    with engine.begin() as conn:
        existing = {
            r[0]: r[1]
            for r in conn.execute(
                text("SELECT obs_date, value FROM prices WHERE series_id = :s"),
                {"s": series_id},
            ).all()
        }
        # normalise keys to date objects for comparison
        existing = {
            (k if not isinstance(k, str) else datetime.fromisoformat(k).date()): v
            for k, v in existing.items()
        }

        added = updated = 0
        payload = []
        for d, v in rows:
            if d not in existing:
                added += 1
            elif abs(existing[d] - v) > 1e-9:
                updated += 1
            else:
                continue  # unchanged, skip the write entirely
            payload.append({"s": series_id, "d": d, "v": float(v), "t": now})

        if payload:
            upd = ("ON CONFLICT (series_id, obs_date) DO UPDATE SET "
                   "value=EXCLUDED.value, ingested_at=EXCLUDED.ingested_at") if is_postgres(engine) else \
                  ("ON CONFLICT(series_id, obs_date) DO UPDATE SET "
                   "value=excluded.value, ingested_at=excluded.ingested_at")
            conn.execute(
                text("INSERT INTO prices (series_id, obs_date, value, ingested_at) "
                     "VALUES (:s, :d, :v, :t) " + upd),
                payload,
            )
    return added, updated


def log_ingest(engine, source: str, status: str, seen=0, added=0, updated=0, message="") -> None:
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO ingest_log (source, run_at, status, rows_seen, rows_added, "
                 "rows_updated, message) VALUES (:src, :t, :st, :seen, :add, :upd, :msg)"),
            {"src": source, "t": utcnow(), "st": status, "seen": seen,
             "add": added, "upd": updated, "msg": message[:1000]},
        )
