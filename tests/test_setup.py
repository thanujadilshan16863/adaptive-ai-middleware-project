import json

import pandas as pd
import pytest

from backend.app.config import Settings, settings
from backend.app.services.phase2_llm_mapper import HeaderOnlyLLMMapper
from check_setup import check_llm, main


def test_live_required_does_not_silently_fallback(monkeypatch):
    monkeypatch.setattr(settings, 'llm_required', True)
    with pytest.raises(ValueError, match='LLM_API_KEY'):
        HeaderOnlyLLMMapper().map(pd.DataFrame({'sku': ['A']}))
    assert HeaderOnlyLLMMapper(offline=True).map(pd.DataFrame({'sku': ['A']})).backend == 'offline_header_rules'


def test_smoke_test_rejects_missing_key():
    with pytest.raises(ValueError, match='LLM_API_KEY'):
        check_llm()


def test_configuration_validation():
    with pytest.raises(ValueError):
        Settings(_env_file=None, forecast_mode='typo')
    with pytest.raises(ValueError):
        Settings(_env_file=None, chronos_batch_size=0)


def test_default_setup_does_not_call_services(monkeypatch, tmp_path):
    import check_setup
    def forbidden():
        raise AssertionError('Default preflight must not call any external service')
    monkeypatch.setattr(check_setup, 'check_llm', forbidden)
    monkeypatch.setattr(check_setup, 'check_chronos', forbidden)
    monkeypatch.setattr(check_setup, 'check_database', forbidden)
    output = tmp_path / 'checks.json'
    monkeypatch.setattr('sys.argv', ['check_setup.py', '--output', str(output)])
    assert main() == 0
    report = json.loads(output.read_text())
    assert all(check['status'] == 'not_run' for check in report['checks'].values())
