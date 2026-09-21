"""Two ways to get a schema - create_all for an empty database, migrations for
an existing one. They must not drift apart, and adopting a database that
predates migrations must never cost it a row."""

from __future__ import annotations

import sqlite3

import pytest
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
    with sqlite3.connect(backup) as con:  # the backup is the database as it was, and usable
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "alembic_version" not in tables
        assert con.execute("SELECT name FROM organizations").fetchall() == [("A",)]
        assert con.execute("PRAGMA integrity_check").fetchone() == ("ok",)

    db.init_db()  # and from now on startup upgrades it like any other
    assert ("organization_id", "sku") in schema(url)["garments"]["unique"]


def test_upgrade_is_a_no_op_at_head():
    db.init_db()
    before = schema(get_settings().database_url)
    db.upgrade_db()
    assert schema(get_settings().database_url) == before


def _legacy_database() -> str:
    """A database as `create_all` left it before migrations existed."""
    command.upgrade(db.alembic_config(), db.BASELINE)
    path = get_settings().database_url.split("///", 1)[-1]
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE alembic_version")
    db.get_engine().dispose()
    return path


def _add_run(con, run_id: str, experiment: str, status: str, created: str) -> None:
    con.execute(
        "INSERT INTO experiment_runs (id, organization_id, experiment_id, status, seed, code_version,"
        " detector_version, dataset_version, configuration, error, log, created_at)"
        " VALUES (?, 'o', ?, ?, 1, '', '', 1, '{}', '', '[]', ?)", (run_id, experiment, status, created))


def test_adoption_closes_the_duplicate_active_runs_the_new_index_forbids():
    path = _legacy_database()
    with sqlite3.connect(path) as con:
        _add_run(con, "old", "e1", "running", "2026-01-01")
        _add_run(con, "new", "e1", "queued", "2026-01-02")
        _add_run(con, "solo", "e2", "running", "2026-01-01")
        _add_run(con, "done", "e1", "completed", "2026-01-03")

    db.upgrade_db()
    with sqlite3.connect(path) as con:
        status = dict(con.execute("SELECT id, status FROM experiment_runs"))
        assert status == {"old": "failed", "new": "queued", "solo": "running", "done": "completed"}
        assert "superseded" in con.execute("SELECT error FROM experiment_runs WHERE id='old'").fetchone()[0]
    indexes = schema(get_settings().database_url)["experiment_runs"]["indexes"]
    assert (("experiment_id",), True, "status IN ('queued', 'running')") in indexes  # predicate intact


def test_a_failed_upgrade_leaves_the_database_exactly_as_it_was(monkeypatch):
    """Otherwise the stamp survives, the database reads as 'versioned', and every
    later start of the API retries the same failing upgrade and dies."""
    path = _legacy_database()
    with sqlite3.connect(path) as con:
        _add_run(con, "r", "e1", "completed", "2026-01-01")
    before = schema(get_settings().database_url)

    def broken(config, revision):
        raise RuntimeError("disk full")

    monkeypatch.setattr(db.command, "upgrade", broken)
    with pytest.raises(RuntimeError):
        db.upgrade_db()
    monkeypatch.undo()

    assert db.schema_state() == "legacy" and schema(get_settings().database_url) == before
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT id FROM experiment_runs").fetchall() == [("r",)]
    db.init_db()  # and the API still starts
    db.upgrade_db()  # and a later attempt succeeds
    assert db.schema_state() == "versioned"


def test_downgrade_restores_the_previous_schema():
    config = db.alembic_config()
    command.upgrade(config, db.BASELINE)
    baseline = schema(get_settings().database_url)
    command.upgrade(config, "head")
    command.downgrade(config, db.BASELINE)
    assert schema(get_settings().database_url) == baseline


def test_an_in_memory_database_is_stamped_where_its_tables_are(monkeypatch):
    monkeypatch.setenv("VEIL_DATABASE_URL", "sqlite://")
    get_settings.cache_clear()
    db._engine = None
    db.init_db()
    assert db.schema_state() == "versioned"
    assert len(set(db.revisions())) == 1


def test_upgrade_at_head_makes_no_backup(tmp_path):
    db.init_db()
    assert db.upgrade_db() is None
    assert not list(tmp_path.glob("*.bak-*"))
