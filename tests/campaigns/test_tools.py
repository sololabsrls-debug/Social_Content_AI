from unittest.mock import MagicMock, patch

TENANT = "tenant-123"


def _mock_sb(data):
    """Return a mock Supabase client whose .execute() returns data."""
    sb = MagicMock()
    chain = sb.table.return_value
    for attr in [
        "select",
        "eq",
        "in_",
        "gte",
        "lte",
        "lt",
        "gt",
        "order",
        "limit",
        "is_",
        "filter",
        "neq",
        "ilike",
        "maybe_single",
        "single",
    ]:
        setattr(chain, attr, MagicMock(return_value=chain))
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


def test_propose_campaign_schema_exposes_variant_and_treatment():
    from src.campaigns.tools import TOOL_SCHEMAS

    schema = next(tool for tool in TOOL_SCHEMAS if tool["name"] == "propose_campaign")
    props = schema["input_schema"]["properties"]
    required = schema["input_schema"]["required"]

    assert "objective_summary" in props
    assert "wa_message_variant" in props
    assert "treatment_label" in props
    assert "objective_summary" in required
    assert "wa_message_variant" in required
    assert "treatment_label" in required


def test_build_campaign_graphic_prompt_is_constrained():
    from src.social.gemini_social import build_campaign_graphic_prompt

    prompt = build_campaign_graphic_prompt("Manicure Spa", "Centro Aurora")
    assert "Manicure Spa" in prompt
    assert "Centro Aurora" in prompt
    assert "Nessun altro testo" in prompt
    assert "Non trasformare il messaggio WhatsApp in testo grafico" in prompt


def test_build_campaign_graphic_system_instruction_is_visual_only():
    from src.social.gemini_social import build_campaign_graphic_system_instruction

    tenant = {
        "name": "Centro Aurora",
        "theme_primary_color": "#111111",
        "theme_secondary_color": "#222222",
        "social_profile": {
            "visual_style": "minimal",
            "photo_style": "bright_natural",
            "typography_style": "serif_elegant",
        },
    }
    instruction = build_campaign_graphic_system_instruction(tenant)
    assert "copywriting" in instruction
    assert "CTA standard" not in instruction
