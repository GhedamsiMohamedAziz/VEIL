"""Operator CLI: bootstrap an org, run an experiment, print a report."""

from __future__ import annotations

import argparse
import json
import sys

from alembic import command as alembic

from veil.db import (
    alembic_config,
    create_organization,
    init_db,
    revisions,
    schema_state,
    session_scope,
    upgrade_db,
)
from veil.ml.detectors import registry
from veil.ml.runner import execute_run
from veil.models import Experiment, ExperimentRun


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veil")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create database tables")
    sub.add_parser("detectors", help="list registered detectors")
    db = sub.add_parser("db", help="schema migrations")
    db.add_argument("action", choices=["status", "upgrade", "revision"],
                    help="status: what would change. upgrade: back up (SQLite), then migrate. "
                         "revision: autogenerate a migration from the models")
    db.add_argument("message", nargs="?", default="schema change")

    org = sub.add_parser("create-org", help="create an organization and print its API key once")
    org.add_argument("name")
    org.add_argument("email")
    org.add_argument("--api-key", default=None)

    run = sub.add_parser("run", help="execute an experiment synchronously")
    run.add_argument("experiment_id")
    run.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.command == "init":
        init_db()
        print("database ready")
    elif args.command == "db":
        if args.action == "status":
            state = schema_state()
            current, head = revisions()
            print(f"database is {state}")
            if state == "legacy":
                print(f"current: none - created before migrations; `veil db upgrade` adopts it\nhead:    {head}")
            elif state == "versioned":
                print(f"current: {current}\nhead:    {head}" + ("" if current == head else "   <- pending"))
        elif args.action == "revision":
            alembic.revision(alembic_config(), message=args.message, autogenerate=True)
            print("read it before committing: autogenerate cannot see unnamed constraints")
        else:
            if schema_state() == "versioned" and len(set(revisions())) == 1:
                print("already at head; nothing to do")
            else:
                backup = upgrade_db()
                print(f"backup: {backup}" if backup else
                      "no backup made: not an SQLite file - this safety net is SQLite-only")
                print("database is at head")
    elif args.command == "detectors":
        for info in registry.available():
            print(f"{info.id:28} {info.version:40} differentiable={info.differentiable}")
    elif args.command == "create-org":
        init_db()
        with session_scope() as session:
            _, user, key = create_organization(session, args.name, args.email, args.api_key)
            print(json.dumps({"user_id": user.id, "api_key": key}, indent=2))
            print("Store this key now - it is not recoverable.", file=sys.stderr)
    elif args.command == "run":
        init_db()
        with session_scope() as session:
            experiment = session.get(Experiment, args.experiment_id)
            if experiment is None:
                print("no such experiment", file=sys.stderr)
                return 1
            run_row = ExperimentRun(
                organization_id=experiment.organization_id, experiment_id=experiment.id,
                status="queued", seed=args.seed, configuration=experiment.configuration,
            )
            session.add(run_row)
            session.flush()
            run_id = run_row.id
        print(json.dumps(execute_run(run_id), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
