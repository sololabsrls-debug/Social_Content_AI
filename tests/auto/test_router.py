# tests/auto/test_router.py
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from main import app
    fake_tenant = {"id": "t1", "name": "Test", "display_name": "Test"}

    # Override FastAPI dependency directly
    import src.campaigns.auto.router as auto_router
    import src.campaigns.router as campaigns_router
    app.dependency_overrides[auto_router.get_tenant] = lambda: fake_tenant
    app.dependency_overrides[campaigns_router.get_tenant] = lambda: fake_tenant

    monkeypatch.setattr("src.campaigns.auto.router.get_supabase", lambda: MagicMock())
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_get_config_200(client, monkeypatch):
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {"campaigns_per_month": 4, "is_active": True}
    ]
    monkeypatch.setattr("src.campaigns.auto.router.get_supabase", lambda: sb)
    resp = client.get("/campaigns/auto/config", headers={"X-API-Key": "key"})
    assert resp.status_code == 200
    assert resp.json()["campaigns_per_month"] == 4


def test_approve_campaign_200(client, monkeypatch):
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {"id": "c1", "tenant_id": "t1", "status": "auto_pending"}
    ]
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [{}]
    monkeypatch.setattr("src.campaigns.auto.router.get_supabase", lambda: sb)
    resp = client.put("/campaigns/auto/c1/approve", headers={"X-API-Key": "key"})
    assert resp.status_code == 200


def test_approve_wrong_status_409(client, monkeypatch):
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {"id": "c1", "tenant_id": "t1", "status": "auto_approved"}
    ]
    monkeypatch.setattr("src.campaigns.auto.router.get_supabase", lambda: sb)
    resp = client.put("/campaigns/auto/c1/approve", headers={"X-API-Key": "key"})
    assert resp.status_code == 409
