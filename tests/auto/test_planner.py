# tests/auto/test_planner.py
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
import pytest

from src.campaigns.auto.planner import generate_monthly_plan


def _make_sb():
    sb = MagicMock()
    return sb


def _plan_row(status="generating", started_offset_min=-5):
    started = (datetime.utcnow() + timedelta(minutes=started_offset_min)).isoformat()
    return {"id": "plan-1", "status": status, "generation_started_at": started}


def _bundle(id="b1", sids=None, pids=None):
    return {
        "id": id, "name": "Bundle Test",
        "service_ids": sids or ["s1"],
        "product_ids": pids or ["p1"],
        "bundle_price": 99.0,
    }


def _config(n=4):
    return {"campaigns_per_month": n}


# --- Idempotency: skip if already ready ---
def test_skips_ready_plan(monkeypatch):
    sb = _make_sb()
    # Existing plan: status=ready
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [_plan_row("ready")]
    monkeypatch.setattr("src.campaigns.auto.planner.get_supabase", lambda: sb)

    result = generate_monthly_plan("tenant-1", 6, 2026)
    assert result == "plan-1"
    # Should NOT insert new wa_campaigns
    insert_calls = [c for c in sb.method_calls if "insert" in str(c)]
    assert len(insert_calls) == 0


# --- Retry: reset generating plan after 30min timeout ---
def test_resets_timed_out_generating_plan(monkeypatch):
    sb = _make_sb()
    old_plan = _plan_row("generating", started_offset_min=-35)  # 35 min ago → timeout

    def table_side(name):
        t = MagicMock()
        t.select.return_value.eq.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [old_plan]
        t.update.return_value.eq.return_value.execute.return_value.data = [old_plan]
        t.delete.return_value.eq.return_value.in_.return_value.execute.return_value.data = []
        t.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [_config(2)]
        t.select.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [_bundle("b1"), _bundle("b2", ["s2"], ["p2"])]
        t.insert.return_value.execute.return_value.data = [{"id": "c1"}, {"id": "c2"}]
        return t

    sb.table.side_effect = table_side
    monkeypatch.setattr("src.campaigns.auto.planner.get_supabase", lambda: sb)

    result = generate_monthly_plan("tenant-1", 6, 2026)
    assert result == "plan-1"


# --- Few bundles: mark plan partial ---
def test_marks_partial_when_fewer_bundles_than_requested(monkeypatch):
    sb = _make_sb()

    def table_side(name):
        t = MagicMock()
        if name == "auto_campaign_plans":
            t.select.return_value.eq.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
            t.insert.return_value.execute.return_value.data = [{"id": "plan-new"}]
            t.update.return_value.eq.return_value.execute.return_value.data = [{}]
        elif name == "auto_campaign_configs":
            t.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [_config(4)]
        elif name == "bundles":
            # Only 2 valid bundles available, requested 4
            t.select.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
                _bundle("b1"), _bundle("b2", ["s2"], ["p2"])
            ]
            t.update.return_value.in_.return_value.execute.return_value.data = []
        elif name == "wa_campaigns":
            t.insert.return_value.execute.return_value.data = [{"id": "c1"}, {"id": "c2"}]
        return t

    sb.table.side_effect = table_side
    monkeypatch.setattr("src.campaigns.auto.planner.get_supabase", lambda: sb)

    result = generate_monthly_plan("tenant-1", 6, 2026)
    assert result == "plan-new"
    plan_table_calls = [c for c in sb.table.call_args_list if c[0][0] == "auto_campaign_plans"]
    assert len(plan_table_calls) >= 1


# --- Invalid bundle (wrong array lengths) is skipped ---
def test_skips_invalid_bundle(monkeypatch):
    sb = _make_sb()

    def table_side(name):
        t = MagicMock()
        if name == "auto_campaign_plans":
            t.select.return_value.eq.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
            t.insert.return_value.execute.return_value.data = [{"id": "plan-x"}]
            t.update.return_value.eq.return_value.execute.return_value.data = [{}]
        elif name == "auto_campaign_configs":
            t.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [_config(2)]
        elif name == "bundles":
            # One valid, one with 2 services (invalid)
            t.select.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
                _bundle("b1"),
                {"id": "b-bad", "name": "Bad", "service_ids": ["s1", "s2"], "product_ids": ["p1"], "bundle_price": 50.0},
            ]
            t.update.return_value.in_.return_value.execute.return_value.data = []
        elif name == "wa_campaigns":
            t.insert.return_value.execute.return_value.data = [{"id": "c1"}]
        return t

    sb.table.side_effect = table_side
    monkeypatch.setattr("src.campaigns.auto.planner.get_supabase", lambda: sb)

    result = generate_monthly_plan("tenant-1", 6, 2026)
    assert result == "plan-x"
