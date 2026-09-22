"""Database engine, sessions, and the tenant-scoping helper."""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Select, create_engine, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from veil.config import get_settings
from veil.models import Base, Organization, User

_T = TypeVar("_T")

_engine = None
_Session: sessionmaker[Session] | None = None


def _make_engine():
    settings = get_settings()
    url = settings.database_url
    kwargs: dict = {"future": True}
    if url.startswith("sqlite"):
        # SQLite lives in var/ by default; make sure the directory exists and
        # allow cross-thread use (the background run worker is a thread).
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def get_engine():
    global _engine, _Session
    if _engine is None:
        _engine = _make_engine()
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


BASELINE = "0001"  # the schema every database had before migrations existed


def alembic_config() -> Config:
    """Alembic, driven from code: no alembic.ini, the URL is the app's own."""
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    config.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))
    return config


def _alembic(*steps: tuple) -> None:
    """Run alembic commands - (function, revision) pairs - on OUR engine, in one
    transaction. Our engine: an in-memory SQLite opened by env.py would be a
    different, throwaway database. One transaction: where DDL is transactional
    (PostgreSQL) a failed upgrade leaves nothing behind, not even the stamp."""
    with get_engine().begin() as connection:
        config = alembic_config()
        config.attributes["connection"] = connection
        for function, revision in steps:
            function(config, revision)


def schema_state() -> str:
    """'empty', 'versioned', or 'legacy' (tables, but created before migrations)."""
    tables = set(inspect(get_engine()).get_table_names())
    if "alembic_version" in tables:
        return "versioned"
    return "legacy" if tables else "empty"


def revisions() -> tuple[str | None, str | None]:
    """(current, head)."""
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    with get_engine().connect() as connection:
        return MigrationContext.configure(connection).get_current_revision(), head


def init_db() -> None:
    """Bring the schema up to date.

    An empty database is created from the models and stamped at head (a test in
    tests/test_migrations.py keeps that identical to running the migrations). A
    versioned one is upgraded. A legacy one is left alone: it holds real data,
    and changing it is an explicit act - `veil db upgrade`, which backs it up first.
    """
    lost_race = False
    for attempt in range(10):
        state = schema_state()
        if state == "legacy" and not lost_race:
            logging.getLogger("veil").warning(
                "database predates migrations and was left unchanged; run `veil db upgrade` "
                "(it makes a backup first) to apply the pending schema changes")
            return
        try:
            if state == "empty":
                Base.metadata.create_all(get_engine())
                _alembic((command.stamp, "head"))
                return
            if state == "versioned":
                _alembic((command.upgrade, "head"))
                return
        except (OperationalError, ProgrammingError):
            # Another process (the API booting while `veil create-org` runs) won
            # the race to create the tables.
            if state != "empty" or attempt == 9:
                raise
            lost_race = True
        time.sleep(0.5)  # tables exist but are not stamped yet: the winner is finishing
    raise RuntimeError("database schema was still being created by another process after 5s")


def _sqlite_file() -> Path | None:
    url = make_url(get_settings().database_url)
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        return None
    path = Path(url.database)
    return path if path.is_file() else None


def upgrade_db() -> Path | None:
    """Migrate to head, adopting a legacy database first. Returns the backup
    made of an SQLite file; None when nothing had to change or the database is
    not a file (PostgreSQL: take your own dump first)."""
    state = schema_state()
    current, head = revisions() if state == "versioned" else (None, None)
    if state == "versioned" and current == head:
        return None

    backup, path = None, _sqlite_file()
    if path is not None:
        backup = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d-%H%M%S-%f}")
        # SQLite's own snapshot: consistent even if the API is writing, which a
        # file copy is not. 'x' refuses to overwrite an existing backup.
        with backup.open("xb"):
            pass
        with sqlite3.connect(path) as source, sqlite3.connect(backup) as target:
            source.backup(target)

    steps = [(command.upgrade, "head")]
    if state == "legacy":
        steps.insert(0, (command.stamp, BASELINE))
    try:
        _alembic(*steps)
    except Exception:
        # SQLite DDL is not transactional: a failure can leave the stamp, or a
        # half-rebuilt table, behind - and a stamped database makes every later
        # start retry the same failing upgrade. Put the file back as it was.
        get_engine().dispose()
        if backup is not None:
            with sqlite3.connect(backup) as source, sqlite3.connect(path) as target:
                source.backup(target)
        raise
    get_engine().dispose()  # tables were rebuilt under the pool's connections
    return backup


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _Session is not None
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def scoped(model: type[_T], organization_id: str) -> Select[tuple[_T]]:
    """The only sanctioned way to start a query on tenant data."""
    return select(model).where(model.organization_id == organization_id)  # type: ignore[attr-defined]


def create_organization(session: Session, name: str, email: str, api_key: str | None = None) -> tuple[Organization, User, str]:
    """Create an org plus its first user. Returns the plaintext key once."""
    key = api_key or "veil_" + secrets.token_urlsafe(32)
    org = Organization(name=name)
    session.add(org)
    session.flush()
    user = User(organization_id=org.id, email=email, api_key_hash=hash_api_key(key), role="owner")
    session.add(user)
    session.flush()
    return org, user, key
