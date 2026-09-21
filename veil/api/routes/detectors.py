"""The allowed-detector catalogue (read-only by design)."""

from __future__ import annotations

from fastapi import APIRouter

from veil.api.deps import UserDep
from veil.ml.detectors import registry
from veil.schemas import DetectorOut

router = APIRouter(tags=["detectors"])


@router.get("/detectors", response_model=list[DetectorOut])
def list_detectors(user: UserDep) -> list[dict]:
    """Detectors VEIL may run experiments against.

    The list is fixed in code. There is no endpoint to add one, because that
    would mean loading a user-supplied checkpoint (docs/security.md).
    """
    return [info.as_dict() for info in registry.available()]
