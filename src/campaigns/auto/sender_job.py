# src/campaigns/auto/sender_job.py
import logging
from datetime import datetime, timedelta, timezone

from src.campaigns.message_utils import render_campaign_message
from src.campaigns.wa_sender import send_whatsapp_message
from src.supabase_client import get_supabase

logger = logging.getLogger("AUTO.sender")

SENDING_TIMEOUT_MINUTES = 15


def poll_approved_campaigns() -> None:
    """
    Cron job (every 30 min):
    1. Recover stale 'sending' rows (crash recovery).
    2. Claim auto_approved campaigns with scheduled_at <= now() → sending.
    3. Send WhatsApp to all recipients.
    4. Mark sent or auto_send_error.
    """
    sb = get_supabase()
    _recover_stale_sending(sb)
    _claim_and_send(sb)


def _recover_stale_sending(sb) -> None:
    """Reset 'sending' rows older than SENDING_TIMEOUT_MINUTES → auto_approved."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=SENDING_TIMEOUT_MINUTES)).isoformat()
    stale = sb.table("wa_campaigns").select("id, tenant_id, sending_started_at") \
        .eq("status", "sending") \
        .lt("sending_started_at", cutoff) \
        .execute()
    for row in (stale.data or []):
        logger.warning("Recovering stale sending campaign %s", row["id"])
        sb.table("wa_campaigns").update({"status": "auto_approved", "sending_started_at": None}) \
            .eq("id", row["id"]).execute()


def _claim_and_send(sb) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()

    # Atomic claim: auto_approved → sending (set sending_started_at)
    claimed = sb.table("wa_campaigns").update({
        "status": "sending",
        "sending_started_at": now_iso,
    }).eq("status", "auto_approved").lte("scheduled_at", now_iso).execute()

    rows = claimed.data or []
    if not rows:
        return

    logger.info("Sender: sending %d auto-approved campaign(s)", len(rows))
    for row in rows:
        try:
            _send_campaign(row, sb)
            sb.table("wa_campaigns").update({
                "status": "sent",
                "sent_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", row["id"]).execute()
            logger.info("Auto campaign %s sent", row["id"])
        except Exception as exc:
            logger.error("Auto campaign %s send failed: %s", row["id"], exc)
            sb.table("wa_campaigns").update({
                "status": "auto_send_error",
                "auto_error_message": str(exc),
            }).eq("id", row["id"]).execute()


def _send_campaign(row: dict, sb) -> None:
    """Send WhatsApp messages to all recipients in target_summary."""
    target_summary = row.get("target_summary") or {}
    client_data = target_summary.get("client_data") or []
    message_template = row.get("message_text") or ""

    if not client_data:
        raise RuntimeError("No client_data in target_summary")

    for client in client_data:
        phone = client.get("phone") or client.get("whatsapp_phone")
        name = client.get("name") or ""
        if not phone:
            continue
        rendered = render_campaign_message(message_template, name)
        import asyncio
        result = asyncio.run(send_whatsapp_message(phone, rendered, row["tenant_id"]))
        if not result.get("ok"):
            raise RuntimeError(f"WA send to {phone} failed: {result.get('error', 'unknown')}")
