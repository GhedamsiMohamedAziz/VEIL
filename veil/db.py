"""Database engine, sessions, and the tenant-scoping helper."""

from __future__ import annotations

import hashlib
import logging
import secrets
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from alembic import command
from alembic.config import Config
from sqlalchemy import Select, create_engine, inspect, select
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


def schema_state() -> str:
    """'empty', 'versioned', or 'legacy' (tables, but created before migrations)."""
    tables = set(inspect(get_engine()).get_table_names())
    if "alembic_version" in tables:
        return "versioned"
    return "legacy" if tables else "empty"


def init_db() -> None:
    """Bring the schema up to date.

    An empty database is created from the models and stamped at head (a test in
    tests/test_migrations.py keeps that identical to running the migrations). A
    versioned one is upgraded. A legacy one is left alone: it holds real data,
    and changing it is an explicit act - `veil db upgrade`, which backs it up first.
    """
    state = schema_state()
    if state == "empty":
        Base.metadata.create_all(get_engine())
        command.stamp(alembic_config(), "head")
    elif state == "versioned":
        command.upgrade(alembic_config(), "head")
    else:
        logging.getLogger("veil").warning(
            "database predates migrations and was left unchanged; run `veil db upgrade` "
            "(it makes a backup first) to apply the pending schema changes")


def upgrade_db() -> Path | None:
    """Migrate to head, adopting a legacy database first. Returns the backup
    made of an SQLite file, or None when there was nothing to back up."""
    backup = None
    url = get_settings().database_url
    path = Path(url.split("///", 1)[-1]) if url.startswith("sqlite") else None
    if path is not None and path.is_file():
        backup = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(path, backup)
    if schema_state() == "legacy":
        command.stamp(alembic_config(), BASELINE)
    command.upgrade(alembic_config(), "head")
    global _engine, _Session
    if _engine is not None:  # tables were rebuilt under it
        _engine.dispose()
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
