"""Alembic environment. Driven from code (`veil.db.alembic_config`), not from
an alembic.ini: the database URL is the application's own setting."""

from alembic import context
from sqlalchemy import create_engine

from veil.config import get_settings
from veil.models import Base

url = context.config.get_main_option("sqlalchemy.url") or get_settings().database_url
with create_engine(url).connect() as connection:
    # render_as_batch: SQLite cannot alter a constraint in place, Alembic
    # rebuilds the table instead. Harmless on PostgreSQL.
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
