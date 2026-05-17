# src/campaigns/auto/public_router.py
"""
Endpoint pubblici per link WhatsApp (senza X-API-Key).
Accesso solo con token nel path. Verificati via X-Public-Key header.
"""
import logging
import os
import threading
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException

from src.campaigns.auto.models import ProposalSelectIn, PublicMessageUpdateIn, PublicTargetUpdateIn, PublicImageUploadIn
from src.campaigns.auto.tokens import validate_token
from src.supabase_client import get_supabase

logger = logging.getLogger("AUTO.public_router")

router = APIRouter(prefix="/public", tags=["public-campaigns"])

PUBLIC_KEY = os.getenv("PUBLIC_CAMPAIGN_KEY", "")


def _check_public_key(x_public_key: str = Header(..., alias="X-Public-Key")) -> None:
    if not PUBLIC_KEY or x_public_key != PUBLIC_KEY:
        raise HTTPException(status_code=401, detail="Accesso non autorizzato")


# ── Selezione proposte ──────────────────────────────────────────────────────

@router.get("/selection/{token}")
async def get_selection(token: str):
    _check_public_key.__wrapped__ if hasattr(_check_public_key, "__wrapped__") else None
    ctx = validate_token(token, "monthly_selection")
    plan_id = ctx["resource_id"]
    sb = get_supabase()

    plan_res = sb.table("auto_campaign_plans").select("month, year, proposals_count, selected_count, selection_confirmed_at") \
        .eq("id", plan_id).limit(1).execute()
    plan = (plan_res.data or [None])[0]
    if not plan:
        raise HTTPException(status_code=404, detail="Piano non trovato")

    # Config per max selezionabili
    config_res = sb.table("auto_campaign_configs").select("campaigns_per_month") \
        .eq("tenant_id", ctx["tenant_id"]).eq("is_active", True).limit(1).execute()
    max_select = ((config_res.data or [{}])[0]).get("campaigns_per_month", 4)

    proposals_res = sb.table("auto_campaign_proposals").select("*") \
        .eq("plan_id", plan_id).order("created_at").execute()

    return {
        "plan": plan,
        "max_selectable": max_select,
        "proposals": proposals_res.data or [],
        "already_confirmed": bool(plan.get("selection_confirmed_at")),
    }


@router.post("/selection/{token}/confirm")
async def confirm_selection(token: str, body: ProposalSelectIn):
    ctx = validate_token(token, "monthly_selection")
    plan_id = ctx["resource_id"]
    tenant_id = ctx["tenant_id"]
    sb = get_supabase()

    # Blocca se già confermato
    plan_check = sb.table("auto_campaign_plans").select("selection_confirmed_at") \
        .eq("id", plan_id).limit(1).execute()
    if (plan_check.data or [{}])[0].get("selection_confirmed_at"):
        raise HTTPException(status_code=409, detail="Selezione già confermata")

    # Verifica max_selectable
    config_res = sb.table("auto_campaign_configs").select("campaigns_per_month") \
        .eq("tenant_id", tenant_id).eq("is_active", True).limit(1).execute()
    max_select = ((config_res.data or [{}])[0]).get("campaigns_per_month", 4)

    if len(body.proposal_ids) > max_select:
        raise HTTPException(
            status_code=422,
            detail=f"Puoi selezionare al massimo {max_select} proposte"
        )

    # Verifica tutte le proposal appartengono al piano del tenant
    props_res = sb.table("auto_campaign_proposals").select("id") \
        .eq("plan_id", plan_id).eq("tenant_id", tenant_id).execute()
    valid_ids = {p["id"] for p in (props_res.data or [])}
    for pid in body.proposal_ids:
        if pid not in valid_ids:
            raise HTTPException(status_code=422, detail=f"Proposta {pid} non valida")

    # Aggiorna status proposte
    now_iso = datetime.now(timezone.utc).isoformat()
    sb.table("auto_campaign_proposals").update({"status": "selected", "selected_at": now_iso}) \
        .in_("id", body.proposal_ids).execute()
    reject_ids = [p["id"] for p in (props_res.data or []) if p["id"] not in set(body.proposal_ids)]
    if reject_ids:
        sb.table("auto_campaign_proposals").update({"status": "rejected"}) \
            .in_("id", reject_ids).execute()

    # Crea wa_campaigns per le proposte selezionate
    plan_res = sb.table("auto_campaign_plans").select("month, year") \
        .eq("id", plan_id).limit(1).execute()
    plan = (plan_res.data or [{}])[0]
    month, year = plan.get("month", datetime.now().month), plan.get("year", datetime.now().year)

    selected_proposals_res = sb.table("auto_campaign_proposals").select("*") \
        .in_("id", body.proposal_ids).execute()
    selected_proposals = selected_proposals_res.data or []

    from src.campaigns.auto.planner import schedule_proposals_as_campaigns
    campaign_ids = schedule_proposals_as_campaigns(sb, tenant_id, plan_id, selected_proposals, month, year)

    # Aggiorna piano
    sb.table("auto_campaign_plans").update({
        "selection_confirmed_at": now_iso,
        "selected_count": len(body.proposal_ids),
    }).eq("id", plan_id).execute()

    # Genera contenuto in background
    tenant_res = sb.table("tenants").select("*").eq("id", tenant_id).limit(1).execute()
    tenant = (tenant_res.data or [{}])[0]

    def _bg():
        from src.campaigns.auto.generator import generate_for_campaigns
        from src.campaigns.auto.scheduler import _finalize_plan_status
        try:
            generate_for_campaigns(campaign_ids, tenant)
            _finalize_plan_status(get_supabase(), plan_id)
        except Exception as exc:
            logger.error("Background generation failed after selection: %s", exc)

    threading.Thread(target=_bg, daemon=True).start()

    return {"ok": True, "campaigns_created": len(campaign_ids)}


# ── Review campagna ─────────────────────────────────────────────────────────

@router.get("/review/{token}")
async def get_review(token: str):
    ctx = validate_token(token, "campaign_review")
    campaign_id = ctx["resource_id"]
    sb = get_supabase()
    res = sb.table("wa_campaigns").select(
        "id, tenant_id, status, objective, message_text, image_url, scheduled_at, "
        "target_summary, auto_bundle_id, auto_error_message, approval_deadline_at"
    ).eq("id", campaign_id).eq("tenant_id", ctx["tenant_id"]).limit(1).execute()
    row = (res.data or [None])[0]
    if not row:
        raise HTTPException(status_code=404, detail="Campagna non trovata")
    return row


@router.patch("/review/{token}/message")
async def update_message(token: str, body: PublicMessageUpdateIn):
    ctx = validate_token(token, "campaign_review")
    campaign_id = ctx["resource_id"]
    sb = get_supabase()
    sb.table("wa_campaigns").update({"message_text": body.message_text}) \
        .eq("id", campaign_id).eq("tenant_id", ctx["tenant_id"]).execute()
    return {"ok": True}


@router.post("/review/{token}/regenerate-image")
async def regenerate_image_public(token: str):
    ctx = validate_token(token, "campaign_review")
    campaign_id = ctx["resource_id"]
    sb = get_supabase()
    row_res = sb.table("wa_campaigns").select("tenant_id") \
        .eq("id", campaign_id).limit(1).execute()
    row = (row_res.data or [None])[0]
    if not row:
        raise HTTPException(status_code=404, detail="Campagna non trovata")
    from src.campaigns.auto.router import _campaign_or_404
    import asyncio
    from src.campaigns.auto.generator import _generate_image
    campaign_full = _campaign_or_404(sb, campaign_id, ctx["tenant_id"])
    tenant_res = sb.table("tenants").select("*").eq("id", ctx["tenant_id"]).limit(1).execute()
    tenant = (tenant_res.data or [{}])[0]
    image_url = asyncio.run(_generate_image(campaign_id, tenant, campaign_full))
    if image_url:
        sb.table("wa_campaigns").update({"image_url": image_url}).eq("id", campaign_id).execute()
        return {"image_url": image_url}
    raise HTTPException(status_code=500, detail="Generazione immagine fallita")


@router.patch("/review/{token}/target")
async def update_target(token: str, body: PublicTargetUpdateIn):
    ctx = validate_token(token, "campaign_review")
    campaign_id = ctx["resource_id"]
    sb = get_supabase()
    sb.table("wa_campaigns").update({"target_summary": body.target_summary}) \
        .eq("id", campaign_id).eq("tenant_id", ctx["tenant_id"]).execute()
    return {"ok": True}


@router.post("/review/{token}/upload-image")
async def upload_image_public(token: str, body: PublicImageUploadIn):
    import base64, uuid, mimetypes
    ctx = validate_token(token, "campaign_review")
    campaign_id = ctx["resource_id"]
    sb = get_supabase()

    row_res = sb.table("wa_campaigns").select("tenant_id") \
        .eq("id", campaign_id).limit(1).execute()
    if not (row_res.data or []):
        raise HTTPException(status_code=404, detail="Campagna non trovata")

    ext = mimetypes.guess_extension(body.mime_type) or ".jpg"
    if ext == ".jpe":
        ext = ".jpg"
    path = f"campaigns/{campaign_id}/{uuid.uuid4()}{ext}"
    image_bytes = base64.b64decode(body.image_data)
    sb.storage.from_("social-media").upload(path, image_bytes, {"content-type": body.mime_type})
    image_url = sb.storage.from_("social-media").get_public_url(path)
    sb.table("wa_campaigns").update({"image_url": image_url}) \
        .eq("id", campaign_id).execute()
    return {"image_url": image_url}


@router.delete("/review/{token}/image")
async def remove_image_public(token: str):
    ctx = validate_token(token, "campaign_review")
    campaign_id = ctx["resource_id"]
    sb = get_supabase()
    sb.table("wa_campaigns").update({"image_url": None}) \
        .eq("id", campaign_id).eq("tenant_id", ctx["tenant_id"]).execute()
    return {"ok": True}


@router.post("/review/{token}/approve")
async def approve_public(token: str):
    ctx = validate_token(token, "campaign_review")
    campaign_id = ctx["resource_id"]
    sb = get_supabase()
    row = sb.table("wa_campaigns").select("status, approval_deadline_at") \
        .eq("id", campaign_id).eq("tenant_id", ctx["tenant_id"]).limit(1).execute()
    campaign = (row.data or [None])[0]
    if not campaign:
        raise HTTPException(status_code=404, detail="Campagna non trovata")
    if campaign["status"] not in ("auto_pending", "auto_draft"):
        raise HTTPException(status_code=409, detail=f"Campagna in stato '{campaign['status']}', non approvabile")
    # Controlla deadline
    if campaign.get("approval_deadline_at"):
        deadline = datetime.fromisoformat(campaign["approval_deadline_at"].replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        if deadline < datetime.now(timezone.utc):
            raise HTTPException(status_code=410, detail="Deadline approvazione scaduta")
    sb.table("wa_campaigns").update({"status": "auto_approved"}).eq("id", campaign_id).execute()
    return {"ok": True}


@router.post("/review/{token}/reject")
async def reject_public(token: str):
    ctx = validate_token(token, "campaign_review")
    campaign_id = ctx["resource_id"]
    sb = get_supabase()
    row = sb.table("wa_campaigns").select("status") \
        .eq("id", campaign_id).eq("tenant_id", ctx["tenant_id"]).limit(1).execute()
    campaign = (row.data or [None])[0]
    if not campaign:
        raise HTTPException(status_code=404, detail="Campagna non trovata")
    if campaign["status"] not in ("auto_pending", "auto_draft", "auto_approved"):
        raise HTTPException(status_code=409, detail=f"Campagna in stato '{campaign['status']}', non rifiutabile")
    sb.table("wa_campaigns").update({"status": "auto_rejected"}).eq("id", campaign_id).execute()
    return {"ok": True}
