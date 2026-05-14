# tests/auto/test_sender_job.py
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone
import pytest

from src.campaigns.auto.sender_job import poll_approved_campaigns


def _approved_row(id="c1"):
    return {
        "id": id, "tenant_id": "t1",
        "message_text": "Ciao {{nome}}!",
        "target_summary": {"client_phones": ["39123"], "client_data": [{"phone": "39123", "name": "Rossi Maria"}]},
        "scheduled_at": "2026-06-07T08:00:00+00:00",
    }


def test_claims_atomically_before_sending(monkeypatch):
    """Status auto_approved → sending before WA send is called."""
    sb = MagicMock()
    claimed = [_approved_row()]
    # chain: update().eq("status","auto_approved").lte().execute()
    sb.table.return_value.update.return_value.eq.return_value.lte.return_value.execute.return_value.data = claimed

    send_calls = []
    monkeypatch.setattr("src.campaigns.auto.sender_job.get_supabase", lambda: sb)
    monkeypatch.setattr("src.campaigns.auto.sender_job._send_campaign", lambda row, sb: send_calls.append(row["id"]))

    poll_approved_campaigns()
    assert "c1" in send_calls


def test_marks_auto_send_error_on_failure(monkeypatch):
    """If _send_campaign raises, status → auto_send_error."""
    sb = MagicMock()
    # chain: update().eq().lte().execute() → claimed row
    sb.table.return_value.update.return_value.eq.return_value.lte.return_value.execute.return_value.data = [_approved_row()]

    monkeypatch.setattr("src.campaigns.auto.sender_job.get_supabase", lambda: sb)
    monkeypatch.setattr("src.campaigns.auto.sender_job._send_campaign",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("WA error")))

    poll_approved_campaigns()
    update_calls = str(sb.table.return_value.update.call_args_list)
    assert "auto_send_error" in update_calls


def test_recovers_stale_sending_rows(monkeypatch):
    """Rows stuck in 'sending' for >15 min are reset to auto_approved."""
    sb = MagicMock()
    # No auto_approved rows to claim
    sb.table.return_value.update.return_value.lte.return_value.execute.return_value.data = []
    # Stale sending rows
    stale_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    sb.table.return_value.select.return_value.eq.return_value.lt.return_value.execute.return_value.data = [
        {"id": "c-stale", "tenant_id": "t1", "sending_started_at": stale_time}
    ]
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [{}]

    monkeypatch.setattr("src.campaigns.auto.sender_job.get_supabase", lambda: sb)

    poll_approved_campaigns()
    update_calls = str(sb.table.return_value.update.call_args_list)
    assert "auto_approved" in update_calls
