# src/campaigns/auto/scheduler.py
import logging
import os
from datetime import datetime, timedelta, timezone

from src.campaigns.auto.notifier import poll_pending_notifications
from src.campaigns.auto.sender_job import poll_approved_campaigns
from src.campaigns.auto.tokens import generate_token
from src.supabase_client import get_supabase

logger = logging.getLogger("AUTO.scheduler")

GESTIONALE_URL_DEFAULT = "https://app.radiantbeauty.it"
GESTIONALE_URL = os.getenv("GESTIONALE_URL", GESTIONALE_URL_DEFAULT)


def run_monthly_proposer() -> None:
    """1° del mese 07:30: genera proposte AI per ogni tenant attivo e invia WA."""
    from src.campaigns.auto.planner import _get_or_create_plan
    from src.campaigns.auto.proposer import generate_proposals
    from src.campaigns.wa_sender import send_platform_message
    import asyncio

    sb = get_supabase()
    now = datetime.now(timezone.utc)
    month, year = now.month, now.year

    res = sb.table("auto_campaign_configs").select("tenant_id").eq("is_active", True).execute()
    tenant_ids = [r["tenant_id"] for r in (res.data or [])]

    for tenant_id in tenant_ids:
        try:
            plan_id, _ = _get_or_create_plan(sb, tenant_id, month, year)
            plan_res = sb.table("auto_campaign_plans").select("selection_link_sent_at") \
                .eq("id", plan_id).limit(1).execute()
            plan = (plan_res.data or [{}])[0]

            if plan.get("selection_link_sent_at"):
                continue  # già inviato

            count = generate_proposals(tenant_id, plan_id, month, year)
            if count == 0:
                logger.warning("No proposals generated for tenant %s", tenant_id)
                continue

            token_raw = generate_token(tenant_id, "monthly_selection", plan_id, expires_days=7)
            link = f"{GESTIONALE_URL}/p/selection/{token_raw}"

            month_names = ["", "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
                           "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"]
            month_name = month_names[month]

            tenant_res = sb.table("tenants").select("owner_phone, phone").eq("id", tenant_id).limit(1).execute()
            tenant_row = (tenant_res.data or [{}])[0]
            owner_phone = tenant_row.get("owner_phone") or tenant_row.get("phone")
            if not owner_phone:
                logger.warning("No owner_phone for tenant %s, skipping WA", tenant_id)
                continue

            text = (
                f"\U0001f338 Ho preparato {count} proposte di campagna per {month_name}!\n\n"
                f"Scegli quelle che ti piacciono:\n{link}\n\n"
                f"Hai 7 giorni per selezionarle."
            )
            asyncio.run(send_platform_message(owner_phone, text))

            sb.table("auto_campaign_plans").update({
                "selection_link_sent_at": now.isoformat()
            }).eq("id", plan_id).execute()

            logger.info("Proposer: sent selection link to tenant %s (%d proposals)", tenant_id, count)
        except Exception as exc:
            logger.error("Proposer failed for tenant %s: %s", tenant_id, exc)


def poll_proposal_reminders() -> None:
    """Ogni 6h: controlla piani senza selezione e invia reminder o fallback."""
    from src.campaigns.wa_sender import send_platform_message
    import asyncio

    sb = get_supabase()
    now = datetime.now(timezone.utc)

    res = sb.table("auto_campaign_plans").select("*") \
        .not_.is_("selection_link_sent_at", "null") \
        .is_("selection_confirmed_at", "null") \
        .is_("fallback_applied_at", "null") \
        .execute()

    for plan in (res.data or []):
        try:
            sent_at = datetime.fromisoformat(plan["selection_link_sent_at"].replace("Z", "+00:00"))
            if sent_at.tzinfo is None:
                sent_at = sent_at.replace(tzinfo=timezone.utc)
            elapsed = now - sent_at

            tenant_res = sb.table("tenants").select("owner_phone, phone").eq("id", plan["tenant_id"]).limit(1).execute()
            tenant_row = (tenant_res.data or [{}])[0]
            phone = tenant_row.get("owner_phone") or tenant_row.get("phone")
            if not phone:
                continue

            # Deadline originale = 7gg dall'invio iniziale
            original_deadline = sent_at + timedelta(days=7)

            if elapsed >= timedelta(days=7) and not plan.get("fallback_applied_at"):
                _apply_fallback(sb, plan)
            elif elapsed >= timedelta(days=4) and not plan.get("reminder_2_sent_at"):
                token_raw = generate_token(
                    plan["tenant_id"], "monthly_selection", plan["id"],
                    expires_days=max(1, (original_deadline - now).days)
                )
                link = f"{GESTIONALE_URL}/p/selection/{token_raw}"
                text = f"\U000023f0 Ultimo promemoria! Scegli le campagne del mese:\n{link}"
                asyncio.run(send_platform_message(phone, text))
                sb.table("auto_campaign_plans").update({"reminder_2_sent_at": now.isoformat()}) \
                    .eq("id", plan["id"]).execute()
                logger.info("Sent reminder 2 to tenant %s", plan["tenant_id"])
            elif elapsed >= timedelta(days=2) and not plan.get("reminder_1_sent_at"):
                token_raw = generate_token(
                    plan["tenant_id"], "monthly_selection", plan["id"],
                    expires_days=max(1, (original_deadline - now).days)
                )
                link = f"{GESTIONALE_URL}/p/selection/{token_raw}"
                text = f"\U0001f4f2 Non dimenticare di scegliere le campagne del mese!\n{link}"
                asyncio.run(send_platform_message(phone, text))
                sb.table("auto_campaign_plans").update({"reminder_1_sent_at": now.isoformat()}) \
                    .eq("id", plan["id"]).execute()
                logger.info("Sent reminder 1 to tenant %s", plan["tenant_id"])
        except Exception as exc:
            logger.error("Reminder failed for plan %s: %s", plan["id"], exc)


def _apply_fallback(sb, plan: dict) -> None:
    """Fallback: genera campagne con bundle fissi (come v1) se estetista non seleziona."""
    from src.campaigns.auto.planner import generate_monthly_plan, _get_config, _get_valid_bundles, _insert_campaign_slots
    from src.campaigns.auto.generator import generate_all_for_plan

    tenant_id = plan["tenant_id"]
    month, year = plan["month"], plan["year"]

    logger.info("Applying fallback for plan %s (tenant %s)", plan["id"], tenant_id)

    config = _get_config(sb, tenant_id)
    if not config:
        logger.warning("No config for tenant %s, fallback skipped", tenant_id)
        return

    bundles = _get_valid_bundles(sb, tenant_id, config["campaigns_per_month"])
    if not bundles:
        logger.warning("No bundles for fallback, tenant %s", tenant_id)
        return

    _insert_campaign_slots(sb, tenant_id, plan["id"], bundles, month, year)

    tenant_res = sb.table("tenants").select("*").eq("id", tenant_id).limit(1).execute()
    tenant = (tenant_res.data or [{}])[0]
    generate_all_for_plan(plan["id"], tenant)
    _finalize_plan_status(sb, plan["id"])

    sb.table("auto_campaign_plans").update({
        "fallback_applied_at": datetime.now(timezone.utc).isoformat(),
        "selection_confirmed_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", plan["id"]).execute()


def _finalize_plan_status(sb, plan_id: str) -> None:
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
    msg = None if status == "ready" else f"{draft_count}/{total} campagne generate"
    sb.table("auto_campaign_plans").update({"status": status, "error_message": msg}).eq("id", plan_id).execute()


def cleanup_expired_proposals() -> None:
    """02:00: archivia proposte scadute non selezionate."""
    sb = get_supabase()
    now_iso = datetime.now(timezone.utc).isoformat()
    sb.table("auto_campaign_proposals").update({"status": "expired"}) \
        .in_("status", ["pending", "rejected"]) \
        .lt("expires_at", now_iso).execute()
    logger.info("Expired proposals archived")


def register_auto_campaign_jobs(scheduler) -> None:
    """Registra tutti i job auto-campagne in APScheduler."""
    # ⚠️ NON aggiungere 'auto_monthly_planner' (v1) — il planner parte solo dopo selezione proposte

    # Proposte il 1° del mese alle 07:30
    scheduler.add_job(
        run_monthly_proposer, trigger="cron", day=1, hour=7, minute=30,
        timezone="Europe/Rome", id="auto_monthly_proposer", replace_existing=True,
    )
    # Reminder ogni 6h
    scheduler.add_job(
        poll_proposal_reminders, trigger="interval", hours=6,
        id="auto_proposal_reminders", replace_existing=True,
    )
    # Notifiche ogni ora
    scheduler.add_job(
        poll_pending_notifications, trigger="interval", hours=1,
        id="auto_notifier", replace_existing=True,
    )
    # Invio campagne ogni 30min
    scheduler.add_job(
        poll_approved_campaigns, trigger="interval", minutes=30,
        id="auto_sender", replace_existing=True,
    )
    # Cleanup proposte scadute ogni notte alle 02:00
    scheduler.add_job(
        cleanup_expired_proposals, trigger="cron", hour=2, minute=0,
        id="auto_proposal_cleanup", replace_existing=True,
    )
    logger.info("Auto campaign jobs registered: proposer, reminders, notifier, sender, cleanup")
