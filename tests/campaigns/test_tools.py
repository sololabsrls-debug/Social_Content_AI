from unittest.mock import MagicMock, patch
from datetime import date, timedelta

TENANT = "tenant-123"


def _mock_sb(data):
    """Return a mock Supabase client whose .execute() returns data."""
    sb = MagicMock()
    chain = sb.table.return_value
    for attr in ["select", "eq", "in_", "gte", "lte", "lt", "gt",
                 "order", "limit", "is_", "filter", "neq", "ilike",
                 "maybe_single", "single"]:
        setattr(chain, attr, MagicMock(return_value=chain))
    # not_ is accessed as a property then chained (.not_.is_(...))
    # so chain.not_ must itself be the chain so chain.not_.is_ works
    chain.not_ = chain
    chain.execute.return_value = MagicMock(data=data)
    return sb


def test_get_clients_overview_returns_counts():
    from src.campaigns.tools import get_clients_overview
    mock_data = [
        {"id": "1", "whatsapp_phone": "+39123", "consent_wa": True},
        {"id": "2", "whatsapp_phone": None, "consent_wa": False},
        {"id": "3", "whatsapp_phone": "+39456", "consent_wa": True},
    ]
    with patch("src.campaigns.tools.get_supabase", return_value=_mock_sb(mock_data)):
        result = get_clients_overview(TENANT)
    assert result["total"] == 3
    assert result["with_wa"] == 2
    assert result["with_consent"] == 2


def test_get_clients_reachable_wa_filters_correctly():
    from src.campaigns.tools import get_clients_reachable_wa
    mock_data = [
        {"id": "1", "name": "Lucia", "whatsapp_phone": "+39111", "consent_wa": True},
    ]
    with patch("src.campaigns.tools.get_supabase", return_value=_mock_sb(mock_data)):
        result = get_clients_reachable_wa(TENANT)
    assert len(result) == 1
    assert result[0]["name"] == "Lucia"


def test_get_services_list_returns_list():
    from src.campaigns.tools import get_services_list
    mock_data = [{"id": "s1", "name": "Manicure", "category_id": None}]
    with patch("src.campaigns.tools.get_supabase", return_value=_mock_sb(mock_data)):
        result = get_services_list(TENANT)
    assert len(result) == 1
    assert result[0]["name"] == "Manicure"
