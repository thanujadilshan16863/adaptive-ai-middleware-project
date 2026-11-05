"""Unit tests must never use a developer's real provider key from .env."""
import pytest


@pytest.fixture(autouse=True)
def isolate_external_services(monkeypatch):
    from backend.app.config import settings
    monkeypatch.setattr(settings, 'llm_api_key', '')
    monkeypatch.setattr(settings, 'llm_required', False)
