def test_derive_canvas_update_target_from_reachable_wa():
    from src.campaigns.agent import derive_canvas_update
    result = [
        {"id": "1", "name": "Lucia", "whatsapp_phone": "+39111"},
        {"id": "2", "name": "Sara", "whatsapp_phone": "+39222"},
    ]
    update = derive_canvas_update("get_clients_reachable_wa", result)
    assert update is not None
    assert update["block"] == "target"
    assert update["data"]["reachable"] == 2


def test_derive_canvas_update_unknown_tool_returns_none():
    from src.campaigns.agent import derive_canvas_update
    update = derive_canvas_update("get_opening_hours", [])
    assert update is None


def test_derive_canvas_update_inactive_clients():
    from src.campaigns.agent import derive_canvas_update
    clients = [{"id": str(i), "name": f"C{i}"} for i in range(5)]
    update = derive_canvas_update("get_inactive_clients", clients)
    assert update is not None
    assert update["block"] == "target"
    assert update["data"]["count"] == 5
