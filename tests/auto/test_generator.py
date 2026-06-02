# tests/auto/test_generator.py
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from src.campaigns.auto.generator import generate_campaign_content


@pytest.fixture
def bundle():
    return {
        "id": "bundle-1",
        "name": "Viso + Siero",
        "service_ids": ["svc-1"],
        "product_ids": ["prod-1"],
        "bundle_price": 89.0,
    }


@pytest.fixture
def campaign_row():
    return {
        "id": "camp-1",
        "tenant_id": "tenant-1",
        "auto_bundle_id": "bundle-1",
        "scheduled_at": "2026-06-07T08:00:00+00:00",
    }


@pytest.fixture
def tenant():
    return {"id": "tenant-1", "name": "Salone Test"}


def test_updates_status_to_auto_draft_on_success(monkeypatch, bundle, campaign_row, tenant):
    """After agent runs successfully, campaign status → auto_draft."""
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {**campaign_row, "message_text": "Ciao {{nome}}!", "status": "ready"}
    ]
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [{}]

    monkeypatch.setattr("src.campaigns.auto.generator.get_supabase", lambda: sb)

    async def fake_agent(messages, tenant_id, campaign_id, **kw):
        yield "done", {"campaign_id": campaign_id}

    monkeypatch.setattr("src.campaigns.auto.generator.run_campaign_agent", fake_agent)
    monkeypatch.setattr("src.campaigns.auto.generator._generate_image", AsyncMock(return_value="http://img.url/x.jpg"))

    import asyncio
    asyncio.run(generate_campaign_content(campaign_row, bundle, tenant))

    update_calls = sb.table.return_value.update.call_args_list
    status_updates = [c for c in update_calls if "auto_draft" in str(c)]
    assert len(status_updates) >= 1


def test_passes_treatment_label_to_image_generation(monkeypatch, bundle, campaign_row, tenant):
    """Regression: the re-select must include target_summary + objective and
    pass them to _generate_image, so the image concept uses the real treatment
    label instead of the generic 'Promozione esclusiva' fallback."""
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {**campaign_row, "message_text": "Ciao {{nome}}!", "status": "ready",
         "target_summary": {"treatment_label": "Pedicure Spa Completo"},
         "objective": "Promuovere Pedicure Spa Completo"}
    ]
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [{}]
    monkeypatch.setattr("src.campaigns.auto.generator.get_supabase", lambda: sb)

    async def fake_agent(messages, tenant_id, campaign_id, **kw):
        yield "done", {"campaign_id": campaign_id}

    monkeypatch.setattr("src.campaigns.auto.generator.run_campaign_agent", fake_agent)

    captured = {}

    async def fake_gen_image(campaign_id, tenant_arg, campaign_arg):
        captured["campaign"] = campaign_arg
        return "http://img/x.jpg"

    monkeypatch.setattr("src.campaigns.auto.generator._generate_image", fake_gen_image)

    import asyncio
    asyncio.run(generate_campaign_content(campaign_row, bundle, tenant))

    # 1. the re-select must request the fields needed for the image concept
    select_cols = [c.args[0] for c in sb.table.return_value.select.call_args_list if c.args]
    assert any("target_summary" in s and "objective" in s for s in select_cols), \
        f"re-select must include target_summary + objective, got {select_cols}"

    # 2. _generate_image must actually receive the treatment label
    assert captured["campaign"].get("target_summary", {}).get("treatment_label") == "Pedicure Spa Completo"
    assert captured["campaign"].get("objective") == "Promuovere Pedicure Spa Completo"


def test_leaves_auto_generating_if_no_message_text(monkeypatch, bundle, campaign_row, tenant):
    """If agent produces no message_text, status stays auto_generating."""
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {**campaign_row, "message_text": None, "status": "ready"}
    ]
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [{}]
    monkeypatch.setattr("src.campaigns.auto.generator.get_supabase", lambda: sb)

    async def fake_agent(messages, tenant_id, campaign_id, **kw):
        yield "done", {"campaign_id": campaign_id}

    monkeypatch.setattr("src.campaigns.auto.generator.run_campaign_agent", fake_agent)
    monkeypatch.setattr("src.campaigns.auto.generator._generate_image", AsyncMock(return_value=None))

    import asyncio
    asyncio.run(generate_campaign_content(campaign_row, bundle, tenant))

    update_calls = str(sb.table.return_value.update.call_args_list)
    assert "auto_draft" not in update_calls
