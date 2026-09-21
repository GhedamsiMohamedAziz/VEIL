"""Background execution of experiment runs.

piggy: an in-process thread pool, not Celery+Redis. A run is minutes of
CPU/GPU-bound torch work on a single box, the queue state already lives in
`experiment_runs.status`, and a restart fails anything left active (api/main.py).
Swap in a real broker when runs must survive a deploy or span machines -
`submit()` is the only seam that changes (ADR-005).
"""

from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor

from veil import events
from veil.config import get_settings

_pool: ThreadPoolExecutor | None = None
_futures: dict[str, Future] = {}


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(
            max_workers=get_settings().max_concurrent_runs, thread_name_prefix="veil-run"
        )
        atexit.register(lambda: _pool and _pool.shutdown(wait=False))
    return _pool


def submit(run_id: str) -> Future:
    from veil.ml.runner import execute_run

    future = _get_pool().submit(execute_run, run_id)
    _futures[run_id] = future
    future.add_done_callback(lambda f: _report(run_id, f))
    return future


def _report(run_id: str, future: Future) -> None:
    _futures.pop(run_id, None)  # or every run ever submitted stays referenced
    exc = future.exception()
    if exc is not None:  # pragma: no cover - execute_run catches its own errors
        events.emit(events.EXPERIMENT_FAILED, run_id=run_id, error=f"{type(exc).__name__}: {exc}")


def wait(run_id: str, timeout: float | None = None) -> None:
    """Block until a submitted run finishes. Used by the CLI and tests."""
    future = _futures.get(run_id)
    if future is not None:
        future.result(timeout=timeout)
