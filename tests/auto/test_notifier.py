# tests/auto/test_notifier.py
from unittest.mock import MagicMock, patch
import pytest

from src.campaigns.auto.notifier import poll_pending_notifications


def _campaign(id="c1", tenant_id="t1", bundle_name="Bundle X", scheduled_at="2026-06-14T08:00:00+00:00"):
    return {
        "id": id, "tenant_id": tenant_id,
        "auto_bundle_id": "b1",
        "scheduled_at": scheduled_at,
    }


def _tenant(id="t1", phone="3901234567"):
    return {"id": id, "phone": phone}


def test_skips_campaigns_without_message_text(monkeypatch):
    """poll_pending_notifications must NOT notify if message_text is NULL (still auto_generating)."""
    sb = MagicMock()
    # UPDATE returns 0 rows (filtered by message_text IS NOT NULL)
    sb.table.return_value.update.return_value.eq.return_value.not_.is_.return_value.lte.return_value.not_.is_.return_value.execute.return_value.data = []
    monkeypatch.setattr("src.campaigns.auto.notifier.get_supabase", lambda: sb)

    wa_calls = []
    monkeypatch.setattr("src.campaigns.auto.notifier._send_whatsapp_link", lambda *a, **kw: wa_calls.append(a))

    poll_pending_notifications()
    assert len(wa_calls) == 0


def _make_sb(claimed):
    """Build a per-table MagicMock for Supabase with correct chain stubs."""
    sb = MagicMock()

    # wa_campaigns table mock
    wa_mock = MagicMock()
    wa_mock.update.return_value.eq.return_value.not_.is_.return_value \
        .lte.return_value.not_.is_.return_value.execute.return_value.data = claimed
    wa_mock.update.return_value.eq.return_value.execute.return_value.data = []

    # tenants table mock
    tenant_mock = MagicMock()
    tenant_mock.select.return_value.eq.return_value.limit.return_value \
        .execute.return_value.data = [_tenant()]

    # bundles table mock — return None so bundle_name stays "Campagna"
    bundle_mock = MagicMock()
    bundle_mock.select.return_value.eq.return_value.limit.return_value \
        .execute.return_value.data = []

    def _table(name):
        if name == "wa_campaigns":
            return wa_mock
        if name == "tenants":
            return tenant_mock
        if name == "bundles":
            return bundle_mock
        return MagicMock()

    sb.table.side_effect = _table
    return sb


def test_claims_auto_notifying_before_sending(monkeypatch):
    """Claim: UPDATE auto_draft→auto_notifying happens BEFORE WhatsApp send."""
    claimed = [{"id": "c1", "tenant_id": "t1", "auto_bundle_id": "b1", "scheduled_at": "2026-06-14T08:00:00+00:00"}]
    sb = _make_sb(claimed)

    monkeypatch.setattr("src.campaigns.auto.notifier.get_supabase", lambda: sb)

    wa_order = []
    def fake_send(phone, campaign_id, bundle_name, tenant_id, scheduled_str, gestionale_url):
        wa_order.append("send")

    monkeypatch.setattr("src.campaigns.auto.notifier._send_whatsapp_link", fake_send)
    monkeypatch.setenv("GESTIONALE_URL", "https://app.test")

    poll_pending_notifications()
    assert "send" in wa_order


def test_sets_auto_notify_error_on_failure(monkeypatch):
    """If WA send raises, campaign goes to auto_notify_error."""
    claimed = [{"id": "c1", "tenant_id": "t1", "auto_bundle_id": "b1", "scheduled_at": "2026-06-14T08:00:00+00:00"}]
    sb = _make_sb(claimed)

    monkeypatch.setattr("src.campaigns.auto.notifier.get_supabase", lambda: sb)
    monkeypatch.setattr("src.campaigns.auto.notifier._send_whatsapp_link",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("WA down")))
    monkeypatch.setenv("GESTIONALE_URL", "https://app.test")

    poll_pending_notifications()

    wa_mock = sb.table("wa_campaigns")
    update_calls = str(wa_mock.update.call_args_list)
    assert "auto_notify_error" in update_calls
