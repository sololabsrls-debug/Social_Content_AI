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


@router.get("/bundles")
async def list_bundles(tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    res = sb.table("bundles").select("id, name, service_ids, product_ids, bundle_price, sort_order, is_active, rotation_used_at") \
        .eq("tenant_id", tenant["id"]).eq("bundle_type", "service_product") \
        .order("sort_order").execute()
    return res.data or []


@router.post("/bundles")
async def create_bundle(body: dict, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    tenant_id = tenant["id"]
    service_ids = body.get("service_ids", [])
    product_ids = body.get("product_ids", [])
    if len(service_ids) != 1 or len(product_ids) != 1:
        raise HTTPException(status_code=422, detail="bundle service_product richiede esattamente 1 servizio e 1 prodotto")

    s_res = sb.table("services").select("id").eq("id", service_ids[0]).eq("tenant_id", tenant_id).limit(1).execute()
    p_res = sb.table("products").select("id").eq("id", product_ids[0]).eq("tenant_id", tenant_id).limit(1).execute()
    if not s_res.data or not p_res.data:
        raise HTTPException(status_code=403, detail="Servizio o prodotto non appartiene al tenant")

    res = sb.table("bundles").insert({
        "tenant_id": tenant_id,
        "bundle_type": "service_product",
        "name": body.get("name", "Nuovo bundle"),
        "service_ids": service_ids,
        "product_ids": product_ids,
        "bundle_price": body.get("bundle_price", 0),
        "is_active": True,
        "sort_order": body.get("sort_order", 0),
    }).execute()
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


@router.put("/{campaign_id}/approve")
async def approve_campaign(campaign_id: str, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    row = _campaign_or_404(sb, campaign_id, tenant["id"])
    if row["status"] != "auto_pending":
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
