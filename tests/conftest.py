"""Test fixtures: every test gets its own database and artifact directory."""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("VEIL_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("VEIL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("VEIL_BOOTSTRAP_API_KEY", "")

    import veil.config as config
    import veil.db as db

    config.get_settings.cache_clear()
    db._engine = None
    db._Session = None
    yield
    config.get_settings.cache_clear()
    db._engine = None
    db._Session = None


@pytest.fixture
def red_png() -> bytes:
    """A solid red square: the colour-blob detector fires on it, which makes
    it a real positive sample rather than a placeholder."""
    array = np.zeros((96, 96, 3), dtype=np.uint8)
    array[:, :, 0] = 230
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from veil.api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def api_key(client):
    """Create an organization directly (there is no public signup endpoint)."""
    from veil.db import create_organization, session_scope

    with session_scope() as session:
        _, _, key = create_organization(session, "Test Org", "test@veil.local")
    return key


@pytest.fixture
def auth(api_key) -> dict[str, str]:
    return {"X-API-Key": api_key}
