"""
Sends WhatsApp messages via the existing Baileys bot.
The bot URL is configured via WA_BOT_URL env var.
"""

import logging
import os
import httpx

logger = logging.getLogger("CAMPAIGNS.wa_sender")

async def send_whatsapp_message(phone: str, message: str, tenant_id: str) -> dict:
    """
    Send a single WhatsApp message via the Baileys bot.
    Returns {"ok": True} or {"ok": False, "error": "..."}.
    """
    wa_bot_url = os.getenv("WA_BOT_URL", "")
    if not wa_bot_url:
        logger.error("WA_BOT_URL not configured")
        return {"ok": False, "error": "WA_BOT_URL not configured"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{wa_bot_url}/send",
                json={"phone": phone, "message": message, "tenant_id": tenant_id},
            )
            resp.raise_for_status()
            return {"ok": True}
    except httpx.HTTPError as exc:
        logger.error("WA send failed for %s: %s", phone, exc)
        return {"ok": False, "error": str(exc)}
