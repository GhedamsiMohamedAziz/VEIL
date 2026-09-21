"""Operator CLI: bootstrap an org, run an experiment, print a report."""

from __future__ import annotations

import argparse
import json
import sys

from veil import jobs
from veil.db import create_organization, init_db, session_scope
from veil.ml.detectors import registry
from veil.ml.runner import execute_run
from veil.models import Experiment, ExperimentRun


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veil")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create database tables")
    sub.add_parser("detectors", help="list registered detectors")

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
