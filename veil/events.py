"""Structured experiment logging.

One JSON line per event, so experiment timelines can be reconstructed from
logs alone. Secrets never pass through here - the caller supplies metadata
explicitly and `emit` refuses reserved key names.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_RESERVED = {"api_key", "password", "token", "secret"}

_logger = logging.getLogger("veil.events")
if not _logger.handlers:  # pragma: no cover - import-time wiring
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.propagate = False

EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
PATTERN_GENERATION_STARTED = "PATTERN_GENERATION_STARTED"
PATTERN_ITERATION_COMPLETED = "PATTERN_ITERATION_COMPLETED"
EVALUATION_STARTED = "EVALUATION_STARTED"
EVALUATION_COMPLETED = "EVALUATION_COMPLETED"
PHYSICAL_TEST_STARTED = "PHYSICAL_TEST_STARTED"
PHYSICAL_TEST_COMPLETED = "PHYSICAL_TEST_COMPLETED"
REPORT_GENERATED = "REPORT_GENERATED"
EXPERIMENT_COMPLETED = "EXPERIMENT_COMPLETED"
EXPERIMENT_FAILED = "EXPERIMENT_FAILED"
AUDIT = "AUDIT"


def emit(event: str, **metadata: Any) -> dict[str, Any]:
    """Log one structured event and return the record (handy for tests)."""
    leaked = _RESERVED & {k.lower() for k in metadata}
    if leaked:
        raise ValueError(f"refusing to log reserved keys: {sorted(leaked)}")
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **metadata,
    }
    _logger.info(json.dumps(record, default=str))
    return record
