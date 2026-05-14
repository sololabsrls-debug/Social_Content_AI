# tests/auto/conftest.py
from unittest.mock import MagicMock
import pytest


def make_execute(data=None, error=None):
    """Return a callable that produces an execute() result."""
    result = MagicMock()
    result.data = data or []
    result.error = error
    return result


@pytest.fixture
def mock_sb(monkeypatch):
    """
    Mock get_supabase() to return a chainable MagicMock.
    Usage in tests: mock_sb.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [...]
    Since MagicMock chains automatically, just configure the final .execute().data.
    """
    sb = MagicMock()
    # Default: all execute() calls return empty list
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    monkeypatch.setattr("src.campaigns.auto.planner.get_supabase", lambda: sb)
    monkeypatch.setattr("src.campaigns.auto.notifier.get_supabase", lambda: sb)
    monkeypatch.setattr("src.campaigns.auto.sender_job.get_supabase", lambda: sb)
    return sb
