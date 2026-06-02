# tests/auto/test_scheduler.py
"""
Regression tests for the silent-WA-failure bug:
the monthly proposer / reminders must NOT mark a plan as 'sent'
when the WhatsApp send fails.
"""
from unittest.mock import MagicMock, AsyncMock

import pytest


def _result(data):
    r = MagicMock()
    r.data = data
    return r


def _chain(data):
    """Mock whose any chain of .eq/.limit/.order/... ends in .execute().data == data."""
    m = MagicMock()
    m.eq.return_value = m
    m.limit.return_value = m
    m.order.return_value = m
    m.in_.return_value = m
    m.is_.return_value = m
    m.not_.is_.return_value = m
    m.execute.return_value = _result(data)
    return m


def _make_sb(updates, *, owner_phone="+393331112233", existing_proposals=2):
    """
    Build a chainable Supabase mock.
    `updates` is a list that collects (table_name, payload) for every .update().
    """
    def table_side_effect(name):
        t = MagicMock()
        if name == "auto_campaign_configs":
            t.select.return_value = _chain([{"tenant_id": "t1"}])
        elif name == "auto_campaign_plans":
            t.select.return_value = _chain([{}])  # no selection_link_sent_at yet

            def _update(payload):
                updates.append((name, payload))
                return _chain([{}])

            t.update.side_effect = _update
        elif name == "auto_campaign_proposals":
            rows = [{"id": f"p{i}"} for i in range(existing_proposals)]
            t.select.return_value = _chain(rows)
        elif name == "tenants":
            t.select.return_value = _chain([{"owner_phone": owner_phone, "phone": ""}])
        else:
            t.select.return_value = _chain([])
        return t

    sb = MagicMock()
    sb.table.side_effect = table_side_effect
    return sb


@pytest.fixture
def patched(monkeypatch):
    """Patch all heavy deps used by run_monthly_proposer / poll_proposal_reminders."""
    monkeypatch.setattr("src.campaigns.auto.planner._get_or_create_plan",
                        lambda sb, tid, m, y: ("plan1", False))
    monkeypatch.setattr("src.campaigns.auto.proposer.generate_proposals",
                        lambda *a, **k: 0)  # should NOT be called (proposals already exist)
    monkeypatch.setattr("src.campaigns.auto.scheduler.generate_token",
                        lambda *a, **k: "tok")


def test_proposer_send_failure_does_not_mark_sent(monkeypatch, patched):
    from src.campaigns.auto import scheduler

    updates = []
    sb = _make_sb(updates)
    monkeypatch.setattr("src.campaigns.auto.scheduler.get_supabase", lambda: sb)
    monkeypatch.setattr("src.campaigns.wa_sender.send_platform_message",
                        AsyncMock(return_value={"ok": False, "error": "Connection Closed"}))

    scheduler.run_monthly_proposer()

    sent_updates = [p for (tbl, p) in updates if "selection_link_sent_at" in p]
    assert sent_updates == [], "selection_link_sent_at must NOT be set when WA send fails"


def test_proposer_send_success_marks_sent(monkeypatch, patched):
    from src.campaigns.auto import scheduler

    updates = []
    sb = _make_sb(updates)
    monkeypatch.setattr("src.campaigns.auto.scheduler.get_supabase", lambda: sb)
    monkeypatch.setattr("src.campaigns.wa_sender.send_platform_message",
                        AsyncMock(return_value={"ok": True}))

    scheduler.run_monthly_proposer()

    sent_updates = [p for (tbl, p) in updates if "selection_link_sent_at" in p]
    assert len(sent_updates) == 1, "selection_link_sent_at must be set exactly once on success"


def test_proposer_reuses_existing_proposals(monkeypatch, patched):
    """Idempotency: generate_proposals must not run when proposals already exist."""
    from src.campaigns.auto import scheduler

    called = {"gen": False}

    def _gen(*a, **k):
        called["gen"] = True
        return 5

    monkeypatch.setattr("src.campaigns.auto.proposer.generate_proposals", _gen)
    updates = []
    sb = _make_sb(updates, existing_proposals=3)
    monkeypatch.setattr("src.campaigns.auto.scheduler.get_supabase", lambda: sb)
    monkeypatch.setattr("src.campaigns.wa_sender.send_platform_message",
                        AsyncMock(return_value={"ok": True}))

    scheduler.run_monthly_proposer()

    assert called["gen"] is False, "must not regenerate proposals when they already exist"
