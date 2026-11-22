"""Boot FinSentinel (unmodified) on the demo's own SQLite database.

Replicates the pattern of ``finsentinel/scripts/run_dev.py``: set env vars,
inject a fake ``core.supabase.connection`` exposing a SQLite engine *before*
any FinSentinel module is imported, then create the tables. The database file
lives under ``demo/.state/`` so Jaivardhan's dev DB is never touched. All
configuration is passed through environment variables — no FinSentinel code
is edited.

Call :func:`boot` before importing anything from FinSentinel.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Mapping
from pathlib import Path

from fldemo import MODEL_VERSION
from fldemo.paths import FINSENTINEL_DIR

_BOOTED: Path | None = None


def base_env(*, operator_token: str, rounds: int, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {
        # Placeholder Postgres settings: never used (the engine is replaced by SQLite).
        "DB_USER": "demo", "DB_PASSWORD": "demo", "DB_HOST": "localhost", "DB_NAME": "demo",
        "ENVIRONMENT": "development",
        "LOG_LEVEL": "INFO",
        "FINSENTINEL_OPERATOR_TOKEN": operator_token,
        "FINSENTINEL_ALLOWED_MODEL_VERSIONS": MODEL_VERSION,
        "FINSENTINEL_ALLOWED_ROUNDS": ",".join(str(r) for r in range(1, rounds + 1)),
        "FINSENTINEL_MAX_UPDATE_AGE_SECONDS": "600",
        "FINSENTINEL_MAX_FUTURE_CLOCK_SKEW_SECONDS": "120",
        "FINSENTINEL_MAX_CONCURRENT_INGESTION": "16",
        # Behind the local proxy/tunnel all devices share 127.0.0.1.
        "FINSENTINEL_INGESTION_RATE_LIMIT_RPM": "6000",
        "FINSENTINEL_BODY_READ_TIMEOUT_SECONDS": "30",
        "FINSENTINEL_SAMPLE_COUNT_TRUST_MODE": "client_claimed",
        "FINSENTINEL_HIGH_RISK_ACTION": "quarantined",
    }
    if extra:
        env.update(extra)
    return env


def boot(db_file: Path, env: Mapping[str, str]) -> None:
    """Configure env + SQLite and register FinSentinel's tables. Idempotent per process."""
    global _BOOTED
    if _BOOTED is not None:
        if _BOOTED != db_file:
            raise RuntimeError(f"FinSentinel already booted on {_BOOTED}")
        return
    if "core.supabase.connection" in sys.modules:
        raise RuntimeError("FinSentinel was imported before fs_boot.boot()")

    os.environ.update(env)
    if str(FINSENTINEL_DIR) not in sys.path:
        sys.path.insert(0, str(FINSENTINEL_DIR))

    from sqlalchemy import create_engine, event

    db_file.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):  # WAL: readers don't block the ingestion writer
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    fake = types.ModuleType("core.supabase.connection")
    fake.engine = engine  # type: ignore[attr-defined]
    sys.modules["core.supabase.connection"] = fake

    import core.supabase.session as session_module
    from core.supabase.base import Base
    from db.models import (  # noqa: F401
        audit_event, audit_outbox, replay_record, trusted_key, trusted_model_schema, update_approval,
    )

    Base.metadata.create_all(engine)
    session_module.SessionLocal.configure(bind=engine)
    _BOOTED = db_file
