# src/campaigns/auto/router.py
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException

from src.campaigns.auto.models import AutoCampaignConfigIn, AutoBundleOrderIn, AutoCampaignRescheduleIn
from src.social.supabase_queries import get_tenant_by_api_key
from src.supabase_client import get_supabase

logger = logging.getLogger("AUTO.router")

router = APIRouter(prefix="/campaigns/auto", tags=["auto-campaigns"])


async def get_tenant(x_api_key: str = Header(..., alias="X-API-Key")) -> dict:
    tenant = get_tenant_by_api_key(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="API key non valida")
    return tenant


def _campaign_or_404(sb, campaign_id: str, tenant_id: str) -> dict:
    res = sb.table("wa_campaigns").select("*") \
        .eq("id", campaign_id).eq("tenant_id", tenant_id).limit(1).execute()
    row = (res.data or [None])[0]
    if not row:
        raise HTTPException(status_code=404, detail="Campagna non trovata")
    return row


@router.post("/plan/{month}/{year}/generate")
async def trigger_generate_plan(month: int, year: int, tenant: dict = Depends(get_tenant)):
    import threading
    from src.campaigns.auto.planner import generate_monthly_plan
    from src.campaigns.auto.generator import generate_all_for_plan
    from src.campaigns.auto.scheduler import _finalize_plan_status

    tenant_id = tenant["id"]
    plan_id = generate_monthly_plan(tenant_id, month, year)

    def _run():
        try:
            sb2 = get_supabase()
            check = sb2.table("auto_campaign_plans").select("status").eq("id", plan_id).limit(1).execute()
            if (check.data or [{}])[0].get("status") != "generating":
                logger.warning("Plan %s not in generating state, skipping background run", plan_id)
                return
            generate_all_for_plan(plan_id, tenant)
            _finalize_plan_status(sb2, plan_id)
        except Exception as exc:
            logger.error("Background generation failed for plan %s: %s", plan_id, exc)

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "plan_id": plan_id}


@router.get("/plan/{month}/{year}")
async def get_plan(month: int, year: int, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    tenant_id = tenant["id"]
    plan_res = sb.table("auto_campaign_plans").select("*") \
        .eq("tenant_id", tenant_id).eq("month", month).eq("year", year) \
        .limit(1).execute()
    plan = (plan_res.data or [None])[0]

    campaigns = []
    if plan:
        c_res = sb.table("wa_campaigns").select(
            "id, status, scheduled_at, notification_due_at, auto_bundle_id, objective, message_text, image_url, auto_error_message"
        ).eq("auto_plan_id", plan["id"]).execute()
        campaigns = c_res.data or []

    return {"plan": plan, "campaigns": campaigns}


@router.get("/config")
async def get_config(tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    res = sb.table("auto_campaign_configs").select("*") \
        .eq("tenant_id", tenant["id"]).eq("is_active", True).limit(1).execute()
    return (res.data or [None])[0] or {"campaigns_per_month": 4, "is_active": False}


@router.post("/config")
async def save_config(body: AutoCampaignConfigIn, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    tenant_id = tenant["id"]
    existing = sb.table("auto_campaign_configs").select("id") \
        .eq("tenant_id", tenant_id).limit(1).execute()
    payload = {"campaigns_per_month": body.campaigns_per_month, "is_active": body.is_active}
    if existing.data:
        sb.table("auto_campaign_configs").update(payload).eq("tenant_id", tenant_id).execute()
    else:
        sb.table("auto_campaign_configs").insert({**payload, "tenant_id": tenant_id}).execute()
    return {"ok": True}


_PROMO_TYPES = (
    "service_product", "service_only", "product_only",
    "multi_session", "service_service", "product_bundle",
    "seasonal", "reactivation",
)


@router.get("/bundles")
async def list_bundles(tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    res = sb.table("bundles").select(
        "id, name, service_ids, product_ids, bundle_price, bundle_type, sort_order, is_active, rotation_used_at"
    ).eq("tenant_id", tenant["id"]).in_("bundle_type", list(_PROMO_TYPES)).order("sort_order").execute()
    return res.data or []


@router.post("/bundles")
async def create_bundle(body: dict, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    tenant_id = tenant["id"]
    bundle_type = body.get("bundle_type", "service_product")
    if bundle_type not in _PROMO_TYPES:
        raise HTTPException(status_code=422, detail=f"bundle_type non valido. Valori accettati: {', '.join(_PROMO_TYPES)}")

    service_ids = body.get("service_ids", [])
    product_ids = body.get("product_ids", [])

    if bundle_type == "service_product":
        if len(service_ids) != 1 or len(product_ids) != 1:
            raise HTTPException(status_code=422, detail="service_product richiede esattamente 1 servizio e 1 prodotto")
    elif bundle_type == "service_only":
        if len(service_ids) != 1:
            raise HTTPException(status_code=422, detail="service_only richiede esattamente 1 servizio")
        product_ids = []
    elif bundle_type == "product_only":
        if len(product_ids) != 1:
            raise HTTPException(status_code=422, detail="product_only richiede esattamente 1 prodotto")
        service_ids = []

    if service_ids:
        s_res = sb.table("services").select("id").eq("id", service_ids[0]).eq("tenant_id", tenant_id).limit(1).execute()
        if not s_res.data:
            raise HTTPException(status_code=403, detail="Servizio non appartiene al tenant")
    if product_ids:
        p_res = sb.table("products").select("id").eq("id", product_ids[0]).eq("tenant_id", tenant_id).limit(1).execute()
        if not p_res.data:
            raise HTTPException(status_code=403, detail="Prodotto non appartiene al tenant")

    insert_data: dict = {
        "tenant_id": tenant_id,
        "bundle_type": bundle_type,
        "name": body.get("name", "Nuova promozione"),
        "service_ids": service_ids,
        "product_ids": product_ids,
        "bundle_price": body.get("bundle_price", 0),
        "is_active": True,
        "sort_order": body.get("sort_order", 0),
    }
    if body.get("original_price") is not None:
        insert_data["original_price"] = body["original_price"]
    if body.get("discount_pct") is not None:
        insert_data["discount_pct"] = body["discount_pct"]

    res = sb.table("bundles").insert(insert_data).execute()
    return res.data[0] if res.data else {}


@router.put("/bundles/{bundle_id}/order")
async def update_bundle_order(bundle_id: str, body: AutoBundleOrderIn, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    sb.table("bundles").update({"sort_order": body.sort_order}) \
        .eq("id", bundle_id).eq("tenant_id", tenant["id"]).execute()
    return {"ok": True}


@router.delete("/bundles/{bundle_id}")
async def delete_bundle(bundle_id: str, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    sb.table("bundles").update({"is_active": False}) \
        .eq("id", bundle_id).eq("tenant_id", tenant["id"]).execute()
    return {"ok": True}


@router.post("/{campaign_id}/regenerate-text")
async def regenerate_campaign_text(campaign_id: str, tenant: dict = Depends(get_tenant)):
    import os
    import anthropic as _anthropic
    import pytz

    sb = get_supabase()
    tenant_id = tenant["id"]

    c_res = sb.table("wa_campaigns").select("auto_bundle_id, scheduled_at") \
        .eq("id", campaign_id).eq("tenant_id", tenant_id).limit(1).execute()
    campaign = (c_res.data or [None])[0]
    if not campaign:
        raise HTTPException(status_code=404, detail="Campagna non trovata")

    bundle: dict = {}
    if campaign.get("auto_bundle_id"):
        b_res = sb.table("bundles").select("name, bundle_price, bundle_type") \
            .eq("id", campaign["auto_bundle_id"]).limit(1).execute()
        bundle = (b_res.data or [None])[0] or {}

    scheduled_str = campaign.get("scheduled_at", "")
    try:
        scheduled_dt = datetime.fromisoformat(scheduled_str.replace("Z", "+00:00"))
        rome_dt = scheduled_dt.astimezone(pytz.timezone("Europe/Rome"))
        date_label = f"{rome_dt.day} {rome_dt.strftime('%B %Y')}"
    except Exception:
        date_label = scheduled_str[:10]

    price_str = f"€{bundle['bundle_price']:.2f}" if bundle.get("bundle_price") else "prezzo speciale"
    bt = bundle.get("bundle_type", "service_product")
    name = bundle.get("name") or "promozione"
    if bt == "service_only":
        promo_desc = f"il trattamento '{name}' a {price_str}"
    elif bt == "product_only":
        promo_desc = f"il prodotto '{name}' a {price_str}"
    else:
        promo_desc = f"il bundle '{name}' a {price_str}"

    tenant_name = tenant.get("name") or tenant.get("display_name") or "il centro"

    try:
        client_obj = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = client_obj.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": (
                    f"Sei una copywriter per centri estetici italiani. "
                    f"Scrivi un messaggio WhatsApp promozionale per {tenant_name} riguardo a {promo_desc}, "
                    f"da inviare il {date_label}. "
                    f"Usa {{{{nome}}}} come segnaposto per il nome della cliente. "
                    f"Tono caldo, personale, max 150 parole. Includi call to action per prenotare. "
                    f"Rispondi SOLO con il testo del messaggio."
                )
            }]
        )
        new_text = msg.content[0].text.strip()
    except Exception as exc:
        logger.error("Regenerate text failed for %s: %s", campaign_id, exc)
        raise HTTPException(status_code=500, detail="Rigenerazione testo fallita")

    sb.table("wa_campaigns").update({"message_text": new_text}).eq("id", campaign_id).execute()
    return {"message_text": new_text}


@router.put("/{campaign_id}/approve")
async def approve_campaign(campaign_id: str, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    row = _campaign_or_404(sb, campaign_id, tenant["id"])
    if row["status"] not in ("auto_pending", "auto_draft"):
        raise HTTPException(status_code=409, detail=f"Campagna in stato '{row['status']}', non approvabile")
    sb.table("wa_campaigns").update({"status": "auto_approved"}) \
        .eq("id", campaign_id).execute()
    return {"ok": True}


@router.put("/{campaign_id}/reject")
async def reject_campaign(campaign_id: str, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    row = _campaign_or_404(sb, campaign_id, tenant["id"])
    if row["status"] not in ("auto_pending", "auto_approved"):
        raise HTTPException(status_code=409, detail=f"Campagna in stato '{row['status']}', non rifiutabile")
    sb.table("wa_campaigns").update({"status": "auto_rejected"}) \
        .eq("id", campaign_id).execute()
    return {"ok": True}


@router.put("/{campaign_id}/reschedule")
async def reschedule_campaign(campaign_id: str, body: AutoCampaignRescheduleIn, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    row = _campaign_or_404(sb, campaign_id, tenant["id"])
    if row["status"] in ("sent", "sending", "auto_send_error"):
        raise HTTPException(status_code=409, detail="Campagna già inviata o in invio")
    sb.table("wa_campaigns").update({"scheduled_at": body.scheduled_at}) \
        .eq("id", campaign_id).execute()
    return {"ok": True}


@router.post("/plan/{month}/{year}/propose")
async def trigger_propose(month: int, year: int, tenant: dict = Depends(get_tenant)):
    """Trigger manuale del proposer AI: genera proposte e invia WA all'owner."""
    import threading
    from datetime import datetime, timezone
    from src.campaigns.auto.planner import _get_or_create_plan
    from src.campaigns.auto.proposer import generate_proposals
    from src.campaigns.auto.tokens import generate_token
    from src.campaigns.wa_sender import send_platform_message
    import asyncio
    import os

    tenant_id = tenant["id"]
    sb = get_supabase()
    plan_id, _ = _get_or_create_plan(sb, tenant_id, month, year)

    # Controlla se il link è già stato inviato
    plan_res = sb.table("auto_campaign_plans").select("selection_link_sent_at") \
        .eq("id", plan_id).limit(1).execute()
    if (plan_res.data or [{}])[0].get("selection_link_sent_at"):
        return {"ok": False, "error": "Link già inviato per questo mese", "plan_id": plan_id}

    def _run():
        try:
            count = generate_proposals(tenant_id, plan_id, month, year)
            if count == 0:
                logger.warning("No proposals generated for tenant %s", tenant_id)
                return

            token_raw = generate_token(tenant_id, "monthly_selection", plan_id, expires_days=7)
            gestionale_url = os.getenv("GESTIONALE_URL", "https://app.radiantbeauty.it")
            link = f"{gestionale_url}/p/selection/{token_raw}"

            month_names = ["", "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
                           "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"]
            month_name = month_names[month] if 1 <= month <= 12 else str(month)

            tenant_res = sb.table("tenants").select("owner_phone, phone").eq("id", tenant_id).limit(1).execute()
            tenant_row = (tenant_res.data or [{}])[0]
            owner_phone = tenant_row.get("owner_phone") or tenant_row.get("phone")

            if owner_phone:
                text = (
                    f"\U0001f338 Ho preparato {count} proposte di campagna per {month_name}!\n\n"
                    f"Scegli quelle che ti piacciono:\n{link}\n\n"
                    f"Hai 7 giorni per selezionarle."
                )
                asyncio.run(send_platform_message(owner_phone, text))
                sb.table("auto_campaign_plans").update({
                    "selection_link_sent_at": datetime.now(timezone.utc).isoformat()
                }).eq("id", plan_id).execute()
                logger.info("Propose+notify: %d proposals, link sent to %s (tenant %s)", count, owner_phone, tenant_id)
            else:
                logger.warning("No owner_phone for tenant %s, WA not sent", tenant_id)
        except Exception as exc:
            logger.error("Propose failed for tenant %s: %s", tenant_id, exc)

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "plan_id": plan_id}


@router.get("/plan/{month}/{year}/proposals")
async def list_proposals(month: int, year: int, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    plan_res = sb.table("auto_campaign_plans").select("id") \
        .eq("tenant_id", tenant["id"]).eq("month", month).eq("year", year).limit(1).execute()
    plan = (plan_res.data or [None])[0]
    if not plan:
        return []
    res = sb.table("auto_campaign_proposals").select("*") \
        .eq("plan_id", plan["id"]).order("created_at").execute()
    return res.data or []


@router.post("/proposals/{proposal_id}/save-as-fixed")
async def save_proposal_as_fixed(proposal_id: str, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    tenant_id = tenant["id"]

    # Fetch proposta
    p_res = sb.table("auto_campaign_proposals").select("*") \
        .eq("id", proposal_id).eq("tenant_id", tenant_id).limit(1).execute()
    proposal = (p_res.data or [None])[0]
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposta non trovata")

    # Fetch piano per month/year
    plan_res = sb.table("auto_campaign_plans").select("month, year") \
        .eq("id", proposal["plan_id"]).limit(1).execute()
    plan = (plan_res.data or [{}])[0]

    # Crea bundle fisso
    bundle_data = {
        "tenant_id": tenant_id,
        "bundle_type": proposal["promo_type"],
        "name": proposal.get("display_name") or proposal["promo_type"],
        "bundle_origin": "ai_monthly",
        "origin_month": plan.get("month"),
        "origin_year": plan.get("year"),
        "promo_config": proposal.get("promo_config", {}),
        "is_active": True,
        "sort_order": 0,
        "bundle_price": (proposal.get("promo_config") or {}).get("bundle_price", 0),
    }
    # Mappa service_ids/product_ids dai promo_config per retrocompatibilità
    config = proposal.get("promo_config") or {}
    if "service_id" in config:
        bundle_data["service_ids"] = [config["service_id"]]
    elif "service_id_1" in config:
        bundle_data["service_ids"] = [config["service_id_1"], config.get("service_id_2", "")]
    else:
        bundle_data["service_ids"] = []
    if "product_id" in config:
        bundle_data["product_ids"] = [config["product_id"]]
    elif "product_ids" in config:
        bundle_data["product_ids"] = config["product_ids"]
    else:
        bundle_data["product_ids"] = []

    sb.table("bundles").insert(bundle_data).execute()
    sb.table("auto_campaign_proposals").update({
        "status": "saved_as_fixed",
        "saved_as_fixed_at": __import__("datetime").datetime.utcnow().isoformat(),
    }).eq("id", proposal_id).execute()

    return {"ok": True}
