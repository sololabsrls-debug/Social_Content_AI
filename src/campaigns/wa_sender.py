"""
WhatsApp senders:
  send_whatsapp_message  — per-tenant bot (campagne ai clienti finali), via WA_BOT_URL
  send_platform_message  — bot piattaforma (notifiche ai titolari), via CENTRAL_BOT_URL
"""

import logging
import os
import httpx

logger = logging.getLogger("CAMPAIGNS.wa_sender")

async def send_whatsapp_message(
    phone: str, message: str, tenant_id: str, image_url: str | None = None
) -> dict:
    """Send via per-tenant Baileys bot. Returns {"ok": True} or {"ok": False, "error": ...}."""
    wa_bot_url = os.getenv("WA_BOT_URL", "")
    if not wa_bot_url:
        logger.error("WA_BOT_URL not configured")
        return {"ok": False, "error": "WA_BOT_URL not configured"}

    wa_api_key = os.getenv("WA_API_KEY", "")
    if not wa_api_key:
        logger.error("WA_API_KEY not configured")
        return {"ok": False, "error": "WA_API_KEY not configured"}

    payload: dict = {"phone": phone.lstrip("+"), "message": message, "tenantId": tenant_id}
    if image_url:
        payload["imageUrl"] = image_url

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{wa_bot_url}/send",
                headers={"X-API-Key": wa_api_key},
                json=payload,
            )
            resp.raise_for_status()
            return {"ok": True}
    except httpx.HTTPError as exc:
        logger.error("WA send failed for %s: %s", phone, exc)
        return {"ok": False, "error": str(exc)}


async def send_platform_message(phone: str, message: str) -> dict:
    """Send via central platform bot (notifiche ai titolari). Returns {"ok": True} or {"ok": False, "error": ...}."""
    central_bot_url = os.getenv("CENTRAL_BOT_URL", "")
    if not central_bot_url:
        logger.error("CENTRAL_BOT_URL not configured")
        return {"ok": False, "error": "CENTRAL_BOT_URL not configured"}

    central_api_key = os.getenv("CENTRAL_BOT_API_KEY", "")
    if not central_api_key:
        logger.error("CENTRAL_BOT_API_KEY not configured")
        return {"ok": False, "error": "CENTRAL_BOT_API_KEY not configured"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{central_bot_url}/send",
                headers={"X-API-Key": central_api_key},
                json={"phone": phone.lstrip("+"), "message": message},
            )
            resp.raise_for_status()
            return {"ok": True}
    except httpx.HTTPError as exc:
        logger.error("Platform WA send failed for %s: %s", phone, exc)
        return {"ok": False, "error": str(exc)}
