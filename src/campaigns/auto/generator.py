# src/campaigns/auto/generator.py
import asyncio
import io
import logging
from datetime import datetime, timezone

import pytz

from src.campaigns.agent import run_campaign_agent
from src.social.gemini_social import generate_campaign_graphic
from src.supabase_client import get_supabase

logger = logging.getLogger("AUTO.generator")


async def generate_campaign_content(campaign: dict, bundle: dict, tenant: dict) -> None:
    """
    Runs the campaign agent for one auto campaign slot.
    The agent calls _save_campaign internally (status → 'ready').
    We then override status to 'auto_draft' if message_text exists, else leave as 'auto_generating'.
    Also generates and saves the campaign image.
    """
    campaign_id = campaign["id"]
    tenant_id = campaign["tenant_id"]

    scheduled_str = campaign.get("scheduled_at", "")
    try:
        scheduled_dt = datetime.fromisoformat(scheduled_str.replace("Z", "+00:00"))
        rome_dt = scheduled_dt.astimezone(pytz.timezone("Europe/Rome"))
        # Safe cross-platform date formatting (%-d is Linux-only)
        date_label = f"{rome_dt.day} {rome_dt.strftime('%B %Y')}"
    except Exception:
        date_label = scheduled_str[:10]

    price_str = f"€{bundle['bundle_price']:.2f}" if bundle.get("bundle_price") else "prezzo speciale"
    bundle_type = bundle.get("bundle_type", "service_product")
    if bundle_type == "service_only":
        promo_desc = f"una promozione sul trattamento '{bundle['name']}' a {price_str}"
    elif bundle_type == "product_only":
        promo_desc = f"una promozione sul prodotto '{bundle['name']}' a {price_str}"
    else:
        promo_desc = f"un pacchetto trattamento + prodotto '{bundle['name']}' a {price_str}"

    prompt = (
        f"Crea una campagna WhatsApp per {promo_desc}. "
        f"Data di invio prevista: {date_label}. "
        f"Analizza il target migliore, poi prepara il messaggio con {{{{nome}}}} come segnaposto."
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        async for event_type, _data in run_campaign_agent(
            messages=messages,
            tenant_id=tenant_id,
            campaign_id=campaign_id,
        ):
            pass  # agent saves to DB internally via _save_campaign
    except Exception as exc:
        logger.error("Agent failed for campaign %s: %s", campaign_id, exc)
        return

    # Check if agent produced message_text
    sb = get_supabase()
    res = sb.table("wa_campaigns").select("message_text").eq("id", campaign_id) \
        .eq("tenant_id", tenant_id).limit(1).execute()
    row = (res.data or [None])[0]

    if not row or not row.get("message_text"):
        logger.warning("Campaign %s: agent produced no message_text, leaving as auto_generating", campaign_id)
        return

    # Generate image
    image_url = await _generate_image(campaign_id, tenant, row)

    updates: dict = {"status": "auto_draft"}
    if image_url:
        updates["image_url"] = image_url

    sb.table("wa_campaigns").update(updates).eq("id", campaign_id).execute()
    logger.info("Campaign %s: auto_draft, image=%s", campaign_id, image_url or "none")


async def _generate_image(campaign_id: str, tenant: dict, campaign: dict) -> str | None:
    try:
        target_summary = campaign.get("target_summary") or {}
        treatment_label = (target_summary.get("treatment_label") or "").strip()
        objective = (campaign.get("objective") or "").strip()
        concept = treatment_label or objective or "Promozione esclusiva"

        feed_bytes, resolved_label = await generate_campaign_graphic(concept, tenant)
        if not feed_bytes:
            return None

        sb = get_supabase()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        path = f"{tenant['id']}/campaigns/{campaign_id}/cover-{ts}.jpg"

        try:
            sb.storage.from_("social-media").upload(
                path=path, file=io.BytesIO(feed_bytes),
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )
        except Exception:
            sb.storage.from_("social-media").upload(
                path=path, file=feed_bytes,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )

        return sb.storage.from_("social-media").get_public_url(path)
    except Exception as exc:
        logger.error("Image generation failed for campaign %s: %s", campaign_id, exc)
        return None


def generate_for_campaigns(campaign_ids: list[str], tenant: dict) -> None:
    """
    Genera contenuto AI per una lista specifica di campaign_id.
    Usato dopo selezione proposte (non genera per tutto il piano).
    """
    sb = get_supabase()
    if not campaign_ids:
        return

    res = sb.table("wa_campaigns") \
        .select("id, tenant_id, auto_bundle_id, auto_proposal_id, scheduled_at, target_summary, objective") \
        .in_("id", campaign_ids) \
        .eq("status", "auto_generating") \
        .execute()
    campaigns = res.data or []

    if not campaigns:
        logger.warning("generate_for_campaigns: no auto_generating campaigns found for ids %s", campaign_ids)
        return

    # Fetch bundles via auto_bundle_id
    bundle_ids = list({c["auto_bundle_id"] for c in campaigns if c.get("auto_bundle_id")})
    bundles_by_id: dict = {}
    if bundle_ids:
        bundles_res = sb.table("bundles").select("id, name, service_ids, product_ids, bundle_price, bundle_type, promo_config") \
            .in_("id", bundle_ids).execute()
        bundles_by_id = {b["id"]: b for b in (bundles_res.data or [])}

    # Fetch proposals for proposal-based campaigns (no auto_bundle_id)
    proposal_ids = list({c["auto_proposal_id"] for c in campaigns
                         if c.get("auto_proposal_id") and not c.get("auto_bundle_id")})
    proposals_as_bundles: dict = {}
    if proposal_ids:
        prop_res = sb.table("auto_campaign_proposals") \
            .select("id, promo_type, promo_config, display_name") \
            .in_("id", proposal_ids).execute()
        for p in (prop_res.data or []):
            config = p.get("promo_config") or {}
            proposals_as_bundles[p["id"]] = {
                "name": p.get("display_name") or p.get("promo_type") or "Promozione",
                "bundle_type": p.get("promo_type", "service_product"),
                "bundle_price": config.get("bundle_price"),
            }

    for campaign in campaigns:
        if campaign.get("auto_bundle_id"):
            bundle = bundles_by_id.get(campaign["auto_bundle_id"]) or {}
        elif campaign.get("auto_proposal_id"):
            bundle = proposals_as_bundles.get(campaign["auto_proposal_id"]) or {}
        else:
            bundle = {}
        # Fallback name from campaign objective
        if not bundle.get("name"):
            bundle = {**bundle, "name": campaign.get("objective") or "Promozione"}
        try:
            asyncio.run(generate_campaign_content(campaign, bundle, tenant))
        except Exception as exc:
            logger.error("generate_for_campaigns failed for %s: %s", campaign["id"], exc)


def generate_all_for_plan(plan_id: str, tenant: dict) -> None:
    """
    Synchronous entry point for APScheduler.
    Fetches all auto_generating campaigns in plan, runs generate_campaign_content for each.
    """
    sb = get_supabase()
    res = sb.table("wa_campaigns") \
        .select("id, tenant_id, auto_bundle_id, scheduled_at, target_summary, objective") \
        .eq("auto_plan_id", plan_id) \
        .eq("status", "auto_generating") \
        .execute()
    campaigns = res.data or []

    if not campaigns:
        logger.warning("No auto_generating campaigns for plan %s", plan_id)
        return

    bundle_ids = list({c["auto_bundle_id"] for c in campaigns if c.get("auto_bundle_id")})
    bundles_res = sb.table("bundles").select("id, name, service_ids, product_ids, bundle_price") \
        .in_("id", bundle_ids).execute()
    bundles_by_id = {b["id"]: b for b in (bundles_res.data or [])}

    for campaign in campaigns:
        bundle = bundles_by_id.get(campaign.get("auto_bundle_id"))
        if not bundle:
            logger.warning("Bundle not found for campaign %s", campaign["id"])
            continue
        try:
            asyncio.run(generate_campaign_content(campaign, bundle, tenant))
        except Exception as exc:
            logger.error("generate_campaign_content failed for %s: %s", campaign["id"], exc)
