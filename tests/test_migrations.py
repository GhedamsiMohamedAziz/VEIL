"""Two ways to get a schema - create_all for an empty database, migrations for
an existing one. They must not drift apart, and adopting a database that
predates migrations must never cost it a row."""

from __future__ import annotations

import sqlite3

from alembic import command
from sqlalchemy import create_engine, inspect

from veil import db
from veil.config import get_settings


def schema(url: str) -> dict:
    inspector, out = inspect(create_engine(url)), {}
    for table in sorted(inspector.get_table_names()):
        if table == "alembic_version":
            continue
        out[table] = {
            "columns": {c["name"]: (str(c["type"]).split("(")[0], bool(c["nullable"]))
                        for c in inspector.get_columns(table)},
            "pk": tuple(inspector.get_pk_constraint(table)["constrained_columns"]),
            "unique": {tuple(sorted(u["column_names"])) for u in inspector.get_unique_constraints(table)},
            "indexes": {(tuple(i["column_names"]), bool(i["unique"]),
                         str(i.get("dialect_options", {}).get("sqlite_where", "")))
                        for i in inspector.get_indexes(table)},
            "fks": {(tuple(f["constrained_columns"]), f["referred_table"])
                    for f in inspector.get_foreign_keys(table)},
        }
    return out


def test_migrations_and_create_all_build_the_same_schema(tmp_path):
    db.init_db()  # empty database: create_all + stamp
    created = schema(get_settings().database_url)
    assert db.schema_state() == "versioned"

    config = db.alembic_config()
    migrated_url = f"sqlite:///{tmp_path}/migrated.db"
    config.set_main_option("sqlalchemy.url", migrated_url)
    command.upgrade(config, "head")
    assert schema(migrated_url) == created
    assert ("organization_id", "sku") in created["garments"]["unique"]
    assert ("sku",) not in created["garments"]["unique"]  # the global constraint is gone


def test_a_database_older_than_migrations_is_adopted_without_losing_a_row(tmp_path, caplog):
    url = get_settings().database_url
    path = url.split("///", 1)[-1]
    config = db.alembic_config()
    command.upgrade(config, db.BASELINE)  # the schema such a database has...
    with sqlite3.connect(path) as con:    # ...without the version table, and with data
        con.execute("DROP TABLE alembic_version")
        con.execute("INSERT INTO organizations (id, name, created_at) "
                    "VALUES ('o1', 'A', '2026-01-01')")
    before = schema(url)

    db.init_db()  # starting the API must not touch it
    assert db.schema_state() == "legacy" and schema(url) == before
    assert "veil db upgrade" in caplog.text

    backup = db.upgrade_db()
    assert backup is not None and backup.is_file()
    assert db.schema_state() == "versioned"
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT name FROM organizations").fetchall() == [("A",)]
        assert con.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    with sqlite3.connect(backup) as con:  # the backup is the database as it was
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "alembic_version" not in tables

    db.init_db()  # and from now on startup upgrades it like any other
    assert ("organization_id", "sku") in schema(url)["garments"]["unique"]


def test_upgrade_is_a_no_op_at_head():
    db.init_db()
    before = schema(get_settings().database_url)
    db.upgrade_db()
    assert schema(get_settings().database_url) == before
