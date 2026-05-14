# src/campaigns/auto/scheduler.py
import logging
from datetime import datetime

from src.campaigns.auto.notifier import poll_pending_notifications
from src.campaigns.auto.sender_job import poll_approved_campaigns
from src.supabase_client import get_supabase

logger = logging.getLogger("AUTO.scheduler")


def run_monthly_planner() -> None:
    """Entry point for APScheduler cron (1st of month, 08:00 Rome)."""
    from src.campaigns.auto.planner import generate_monthly_plan
    from src.campaigns.auto.generator import generate_all_for_plan

    sb = get_supabase()
    now = datetime.utcnow()
    month, year = now.month, now.year

    # Fetch all active configs
    res = sb.table("auto_campaign_configs").select("tenant_id") \
        .eq("is_active", True).execute()
    tenant_ids = [r["tenant_id"] for r in (res.data or [])]

    if not tenant_ids:
        logger.info("Monthly planner: no active tenants")
        return

    logger.info("Monthly planner: processing %d tenant(s) for %d/%d", len(tenant_ids), month, year)

    for tenant_id in tenant_ids:
        try:
            tenant_res = sb.table("tenants").select("id, name, display_name, bio, logo_url, social_profile") \
                .eq("id", tenant_id).limit(1).execute()
            tenant = (tenant_res.data or [None])[0]
            if not tenant:
                logger.warning("Tenant %s not found, skipping", tenant_id)
                continue

            plan_id = generate_monthly_plan(tenant_id, month, year)
            generate_all_for_plan(plan_id, tenant)
            _finalize_plan_status(sb, plan_id)
        except Exception as exc:
            logger.error("Monthly planner failed for tenant %s: %s", tenant_id, exc)


def _finalize_plan_status(sb, plan_id: str) -> None:
    """Mark plan ready/partial based on how many slots reached auto_draft."""
    all_res = sb.table("wa_campaigns").select("status").eq("auto_plan_id", plan_id).execute()
    all_campaigns = all_res.data or []
    if not all_campaigns:
        sb.table("auto_campaign_plans").update({
            "status": "error", "error_message": "Nessuna campagna generata",
        }).eq("id", plan_id).execute()
        return
    draft_count = sum(1 for c in all_campaigns if c["status"] == "auto_draft")
    total = len(all_campaigns)
    status = "ready" if draft_count == total else "partial"
    msg = None if status == "ready" else f"{draft_count}/{total} campagne generate con successo"
    sb.table("auto_campaign_plans").update({"status": status, "error_message": msg}).eq("id", plan_id).execute()
    logger.info("Plan %s finalized: %s (%d/%d drafted)", plan_id, status, draft_count, total)


def register_auto_campaign_jobs(scheduler) -> None:
    """Register all auto campaign cron jobs into the given APScheduler instance."""
    # 1st of each month at 08:00 Europe/Rome
    scheduler.add_job(
        run_monthly_planner,
        trigger="cron",
        day=1, hour=8, minute=0,
        timezone="Europe/Rome",
        id="auto_monthly_planner",
        replace_existing=True,
    )
    # Every hour: send WhatsApp notifications
    scheduler.add_job(
        poll_pending_notifications,
        trigger="interval",
        hours=1,
        id="auto_notifier",
        replace_existing=True,
    )
    # Every 30 min: send approved campaigns
    scheduler.add_job(
        poll_approved_campaigns,
        trigger="interval",
        minutes=30,
        id="auto_sender",
        replace_existing=True,
    )
    logger.info("Auto campaign jobs registered: planner, notifier, sender")
