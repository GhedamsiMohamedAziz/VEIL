"""Alembic environment. Driven from code (`veil.db`), not from an alembic.ini:
the application hands over its own connection, so migrations run on the same
database - and in the same transaction - as the caller. Offline (`--sql`) mode
is not supported."""

from alembic import context
from sqlalchemy import create_engine

from veil.config import get_settings
from veil.models import Base


def run(connection) -> None:
    # render_as_batch: SQLite cannot alter a constraint in place, Alembic
    # rebuilds the table instead. Harmless on PostgreSQL.
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


shared = context.config.attributes.get("connection")
if shared is not None:
    run(shared)
else:  # standalone use, e.g. the tests pointing a config at another file
    url = context.config.get_main_option("sqlalchemy.url") or get_settings().database_url
    with create_engine(url).connect() as own:
        run(own)
        own.commit()
