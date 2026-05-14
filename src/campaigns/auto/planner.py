# src/campaigns/auto/planner.py
import logging
from datetime import datetime, timedelta, timezone

import pytz

from src.supabase_client import get_supabase

logger = logging.getLogger("AUTO.planner")

ROME_TZ = pytz.timezone("Europe/Rome")
SEND_HOUR = 10   # 10:00 Rome time
START_DAY = 7    # First campaign on day 7 of the month
SPREAD_DAYS = 21  # Spread campaigns across 3 weeks


def generate_monthly_plan(tenant_id: str, month: int, year: int) -> str:
    """
    Idempotent. Creates or resets an auto_campaign_plans row and inserts
    wa_campaigns (status=auto_generating) for each selected bundle.
    Returns plan_id.
    """
    sb = get_supabase()
    plan_id, needs_generation = _get_or_create_plan(sb, tenant_id, month, year)
    if not needs_generation:
        return plan_id

    config = _get_config(sb, tenant_id)
    if not config:
        _fail_plan(sb, plan_id, "No active auto_campaign_config found")
        return plan_id

    campaigns_per_month = config["campaigns_per_month"]
    bundles = _get_valid_bundles(sb, tenant_id, campaigns_per_month)

    if not bundles:
        _fail_plan(sb, plan_id, "No valid service_product bundles (need exactly 1 service + 1 product each)")
        return plan_id

    _insert_campaign_slots(sb, tenant_id, plan_id, bundles, month, year)

    # Plan stays 'generating' — scheduler.py calls generate_all_for_plan() next,
    # then _finalize_plan_status() marks it ready/partial after agents complete.
    logger.info("Plan %s for tenant %s %d/%d: %d slots created", plan_id, tenant_id, month, year, len(bundles))
    return plan_id


def _get_or_create_plan(sb, tenant_id: str, month: int, year: int):
    """
    Returns (plan_id, needs_generation).
    - ready/partial → (plan_id, False)
    - generating + NOT timed out → (plan_id, False)
    - generating + timed out, or error → reset row + delete stale auto_generating campaigns → (plan_id, True)
    - missing → insert new row → (plan_id, True)
    """
    res = sb.table("auto_campaign_plans").select("id, status, generation_started_at") \
        .eq("tenant_id", tenant_id).eq("month", month).eq("year", year) \
        .limit(1).execute()
    existing = (res.data or [None])[0]

    if existing:
        if existing["status"] in ("ready", "partial"):
            return existing["id"], False

        if existing["status"] == "generating":
            # Use timezone-aware comparison — Supabase returns +00:00 suffix
            started = datetime.fromisoformat(existing["generation_started_at"].replace("Z", "+00:00"))
            # If naive (no tzinfo), assume UTC
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - started < timedelta(minutes=30):
                logger.info("Plan %s still generating (not timed out)", existing["id"])
                return existing["id"], False
            logger.warning("Plan %s timed out, resetting", existing["id"])

        # Reset existing row (status=generating or status=error or timed-out generating)
        _delete_stale_campaigns(sb, existing["id"])
        sb.table("auto_campaign_plans").update({
            "status": "generating",
            "generation_started_at": datetime.utcnow().isoformat(),
            "error_message": None,
        }).eq("id", existing["id"]).execute()
        return existing["id"], True

    # New plan
    ins = sb.table("auto_campaign_plans").insert({
        "tenant_id": tenant_id,
        "month": month,
        "year": year,
        "status": "generating",
        "generation_started_at": datetime.utcnow().isoformat(),
    }).execute()
    return ins.data[0]["id"], True


def _delete_stale_campaigns(sb, plan_id: str):
    """Delete auto_generating rows from a timed-out/errored plan."""
    sb.table("wa_campaigns").delete() \
        .eq("auto_plan_id", plan_id) \
        .in_("status", ["auto_generating"]) \
        .execute()


def _get_config(sb, tenant_id: str):
    res = sb.table("auto_campaign_configs").select("campaigns_per_month") \
        .eq("tenant_id", tenant_id).eq("is_active", True).limit(1).execute()
    return (res.data or [None])[0]


def _get_valid_bundles(sb, tenant_id: str, limit: int) -> list:
    # Fetch 4x more than needed so filtering invalids doesn't short-change the result.
    res = sb.table("bundles") \
        .select("id, name, service_ids, product_ids, bundle_price") \
        .eq("tenant_id", tenant_id) \
        .eq("bundle_type", "service_product") \
        .eq("is_active", True) \
        .order("rotation_used_at", desc=False, nullsfirst=True) \
        .limit(limit * 4) \
        .execute()
    valid = []
    for b in (res.data or []):
        sids = b.get("service_ids") or []
        pids = b.get("product_ids") or []
        if len(sids) == 1 and len(pids) == 1:
            valid.append(b)
        else:
            logger.warning("Bundle %s skipped: %d services, %d products", b["id"], len(sids), len(pids))
    return valid[:limit]


def _insert_campaign_slots(sb, tenant_id: str, plan_id: str, bundles: list, month: int, year: int):
    n = len(bundles)
    spacing = max(1, SPREAD_DAYS // n)
    rows = []
    bundle_ids = []

    for i, bundle in enumerate(bundles):
        local_dt = datetime(year, month, START_DAY + i * spacing, SEND_HOUR, 0, 0)
        scheduled_utc = ROME_TZ.localize(local_dt).astimezone(pytz.utc)
        notif_local = local_dt - timedelta(days=5)
        # Clamp notification: don't notify before day 1
        if notif_local.day < 1:
            notif_local = notif_local.replace(day=1)
        notif_utc = ROME_TZ.localize(notif_local).astimezone(pytz.utc)

        rows.append({
            "tenant_id": tenant_id,
            "auto_plan_id": plan_id,
            "auto_bundle_id": bundle["id"],
            "status": "auto_generating",
            "scheduled_at": scheduled_utc.isoformat(),
            "notification_due_at": notif_utc.isoformat(),
        })
        bundle_ids.append(bundle["id"])

    if rows:
        sb.table("wa_campaigns").insert(rows).execute()

    # Update rotation timestamps
    sb.table("bundles").update({"rotation_used_at": datetime.utcnow().isoformat()}) \
        .in_("id", bundle_ids).execute()


def _fail_plan(sb, plan_id: str, message: str):
    sb.table("auto_campaign_plans").update({
        "status": "error",
        "error_message": message,
    }).eq("id", plan_id).execute()
    logger.error("Plan %s failed: %s", plan_id, message)
