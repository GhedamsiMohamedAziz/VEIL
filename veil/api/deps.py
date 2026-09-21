"""Authentication and tenant scoping.

Every route that touches user data depends on `current_user`, and every
query it makes goes through `scoped(...)`. There is no code path that reads
a row by id alone.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, TypeVar

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from veil.db import hash_api_key, scoped, session_scope
from veil.models import AuditEvent, User

_T = TypeVar("_T")


def get_session() -> Iterator[Session]:
    with session_scope() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def current_user(
    session: SessionDep,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> User:
    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:]
    if not key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing API key")
    user = session.scalar(select(User).where(User.api_key_hash == hash_api_key(key)))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
    return user


UserDep = Annotated[User, Depends(current_user)]


def fetch(session: Session, model: type[_T], object_id: str, user: User) -> _T:
    """Load one row inside the caller's organization or 404.

    A row in another organization is reported as "not found", never as
    "forbidden" - existence is itself information.
    """
    row = session.scalar(scoped(model, user.organization_id).where(model.id == object_id))  # type: ignore[attr-defined]
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{model.__name__} not found")
    return row


def audit(session: Session, user: User, action: str, target: str = "", **detail) -> None:
    session.add(
        AuditEvent(
            organization_id=user.organization_id, user_id=user.id,
            action=action, target=target, detail=detail,
        )
    )
