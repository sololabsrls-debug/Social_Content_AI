# src/campaigns/auto/notifier.py
import logging
import os
from datetime import datetime, timezone

from src.campaigns.wa_sender import send_platform_message
from src.supabase_client import get_supabase

logger = logging.getLogger("AUTO.notifier")

GESTIONALE_URL = os.getenv("GESTIONALE_URL", "https://app.radiantbeauty.it")


def poll_pending_notifications() -> None:
    """
    Cron job: claim auto_draft campaigns where notification_due_at <= now()
    AND message_text IS NOT NULL, then send WhatsApp approval link.
    Uses auto_notifying as intermediate claim state to prevent double-send.
    """
    sb = get_supabase()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Atomic claim: auto_draft → auto_notifying (only rows with message_text)
    claimed = sb.table("wa_campaigns").update({"status": "auto_notifying"}) \
        .eq("status", "auto_draft") \
        .not_.is_("message_text", "null") \
        .lte("notification_due_at", now_iso) \
        .not_.is_("notification_due_at", "null") \
        .execute()

    rows = claimed.data or []
    if not rows:
        return

    logger.info("Notifier: claiming %d campaign(s) for notification", len(rows))

    for row in rows:
        _notify_one(sb, row)


def _notify_one(sb, campaign: dict) -> None:
    campaign_id = campaign["id"]
    tenant_id = campaign["tenant_id"]

    # Get tenant owner_phone (fallback to phone)
    tenant_res = sb.table("tenants").select("phone, owner_phone, name, display_name") \
        .eq("id", tenant_id).limit(1).execute()
    tenant = (tenant_res.data or [None])[0]
    owner_phone = (tenant or {}).get("owner_phone") or (tenant or {}).get("phone")
    if not tenant or not owner_phone:
        logger.warning("No owner_phone for tenant %s, skipping notification", tenant_id)
        sb.table("wa_campaigns").update({"status": "auto_notify_error", "auto_error_message": "Tenant owner_phone missing"}) \
            .eq("id", campaign_id).execute()
        return

    # Get bundle name
    bundle_name = "Campagna"
    if campaign.get("auto_bundle_id"):
        b_res = sb.table("bundles").select("name").eq("id", campaign["auto_bundle_id"]).limit(1).execute()
        b = (b_res.data or [None])[0]
        if b:
            bundle_name = b["name"]

    scheduled_str = campaign.get("scheduled_at", "")
    try:
        scheduled_dt = datetime.fromisoformat(scheduled_str.replace("Z", "+00:00"))
        import pytz
        rome_dt = scheduled_dt.astimezone(pytz.timezone("Europe/Rome"))
        date_label = f"{rome_dt.day}/{rome_dt.month:02d}"
    except Exception:
        date_label = scheduled_str[:10]

    try:
        from src.campaigns.auto.tokens import generate_token
        expires_days = 3
        if campaign.get("approval_deadline_at"):
            try:
                deadline = datetime.fromisoformat(campaign["approval_deadline_at"].replace("Z", "+00:00"))
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                days_until = max(1, (deadline - datetime.now(timezone.utc)).days)
                expires_days = days_until
            except Exception:
                pass
        token_raw = generate_token(tenant_id, "campaign_review", campaign_id, expires_days=expires_days)
        _send_whatsapp_link(
            phone=owner_phone,
            token=token_raw,
            bundle_name=bundle_name,
            scheduled_str=date_label,
            gestionale_url=GESTIONALE_URL,
        )
        sb.table("wa_campaigns").update({"status": "auto_pending"}) \
            .eq("id", campaign_id).execute()
        logger.info("Notified campaign %s to %s", campaign_id, owner_phone)
    except Exception as exc:
        logger.error("WA send failed for campaign %s: %s", campaign_id, exc)
        sb.table("wa_campaigns").update({
            "status": "auto_notify_error",
            "auto_error_message": str(exc),
        }).eq("id", campaign_id).execute()


def _send_whatsapp_link(
    phone: str, token: str, bundle_name: str, scheduled_str: str, gestionale_url: str
) -> None:
    import asyncio
    link = f"{gestionale_url}/p/review/{token}"
    text = (
        f"\U0001f338 Ho preparato la campagna \"{bundle_name}\" per il {scheduled_str}.\n\n"
        f"Aprila qui per vederla, modificarla e approvarla:\n{link}"
    )
    result = asyncio.run(send_platform_message(phone, text))
    if not result.get("ok"):
        raise RuntimeError(f"WA send failed: {result.get('error', 'unknown')}")
