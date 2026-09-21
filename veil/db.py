"""Database engine, sessions, and the tenant-scoping helper."""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from sqlalchemy import Select, create_engine, select
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


def init_db() -> None:
    """Create tables. piggy: no Alembic yet - single-node, pre-release schema.
    Add migrations before the first external deployment (see ADR-006)."""
    Base.metadata.create_all(get_engine())


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
