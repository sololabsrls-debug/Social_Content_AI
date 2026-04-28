import io
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from src.campaigns.agent import run_campaign_agent
from src.campaigns.message_utils import render_campaign_message
from src.campaigns.models import CampaignChatRequest
from src.campaigns.wa_sender import send_whatsapp_message
from src.social.gemini_social import generate_campaign_graphic
from src.social.supabase_queries import get_tenant_by_api_key
from src.supabase_client import get_supabase

logger = logging.getLogger("CAMPAIGNS.router")

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


async def get_tenant(x_api_key: str = Header(..., alias="X-API-Key")) -> dict:
    tenant = get_tenant_by_api_key(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="API key non valida")
    return tenant


@router.post("/chat")
async def campaign_chat(
    req: CampaignChatRequest,
    tenant: dict = Depends(get_tenant),
):
    """SSE stream: runs Campaign AI agent and emits canvas update events."""
    tenant_id = tenant["id"]

    campaign_id = req.campaign_id
    sb = get_supabase()
    initial_target_count = 0
    existing_client_data: list[dict] = []

    if not campaign_id:
        res = sb.table("wa_campaigns").insert(
            {
                "tenant_id": tenant_id,
                "status": "analyzing",
                "chat_history": [m.model_dump() for m in req.messages],
            }
        ).execute()
        if not res.data:
            logger.error("Failed to create campaign record for tenant %s", tenant_id)
            raise HTTPException(status_code=500, detail="Impossibile creare la campagna")
        campaign_id = res.data[0]["id"]
    else:
        existing = (
            sb.table("wa_campaigns")
            .select("target_summary")
            .eq("id", campaign_id)
            .eq("tenant_id", tenant_id)
            .limit(1)
            .execute()
        )
        if existing.data:
            ts = existing.data[0].get("target_summary") or {}
            phones = ts.get("client_phones") or []
            initial_target_count = ts.get("count") or len(phones)
            existing_client_data = ts.get("client_data") or []
            if not existing_client_data and phones:
                existing_client_data = [{"phone": phone, "name": ""} for phone in phones]

    messages = [m.model_dump() for m in req.messages]

    async def stream():
        try:
            yield f"event: init\ndata: {json.dumps({'campaign_id': campaign_id})}\n\n"
            async for event_type, data in run_campaign_agent(
                messages,
                tenant_id,
                campaign_id,
                initial_target_count,
                existing_client_data,
            ):
                payload = json.dumps(data, default=str)
                yield f"event: {event_type}\ndata: {payload}\n\n"
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/")
async def list_campaigns(tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    res = (
        sb.table("wa_campaigns")
        .select("id, status, objective, recipients_count, sent_at, created_at")
        .eq("tenant_id", tenant["id"])
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    return {"data": res.data or []}


@router.post("/{campaign_id}/send")
async def send_campaign(campaign_id: str, tenant: dict = Depends(get_tenant)):
    """Send the campaign WhatsApp message to all target recipients."""
    sb = get_supabase()
    res = (
        sb.table("wa_campaigns")
        .select("message_text, target_summary, status")
        .eq("id", campaign_id)
        .eq("tenant_id", tenant["id"])
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Campagna non trovata")
    campaign = rows[0]

    message_text = campaign.get("message_text")
    if not message_text:
        raise HTTPException(status_code=400, detail="Messaggio campagna non impostato")

    target_summary = campaign.get("target_summary") or {}
    client_data: list[dict] = target_summary.get("client_data") or []
    if not client_data:
        phones = target_summary.get("client_phones") or []
        client_data = [{"phone": phone, "name": ""} for phone in phones]
    if not client_data:
        raise HTTPException(status_code=400, detail="Nessun destinatario trovato - rianalizza la campagna")

    logger.info("Campaign %s message_text raw: %r", campaign_id, message_text[:200])
    sent = 0
    failed = 0
    for client in client_data:
        personalized = render_campaign_message(message_text, client.get("name", ""))
        logger.info(
            "Sending to %s (name=%r): %r",
            client["phone"],
            client.get("name"),
            personalized[:80],
        )
        result = await send_whatsapp_message(client["phone"], personalized, tenant["id"])
        if result.get("ok"):
            sent += 1
        else:
            failed += 1
            logger.warning("WA send failed to %s: %s", client["phone"], result.get("error"))

    sb.table("wa_campaigns").update(
        {
            "status": "sent",
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "recipients_count": sent,
        }
    ).eq("id", campaign_id).execute()

    return {"sent": sent, "failed": failed, "total": len(client_data)}


@router.post("/{campaign_id}/generate-image")
async def generate_campaign_image(campaign_id: str, tenant: dict = Depends(get_tenant)):
    """Generate a campaign graphic using the campaign treatment label and brand identity."""
    sb = get_supabase()
    res = (
        sb.table("wa_campaigns")
        .select("objective, message_text, reason_text, target_summary")
        .eq("id", campaign_id)
        .eq("tenant_id", tenant["id"])
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Campagna non trovata")
    campaign = rows[0]

    objective = campaign.get("objective") or ""
    message_text = campaign.get("message_text") or ""
    reason_text = campaign.get("reason_text") or ""
    target_summary = campaign.get("target_summary") or {}
    treatment_label = (target_summary.get("treatment_label") or "").strip()

    concept_source = (
        treatment_label
        or objective
        or reason_text
        or message_text[:120]
        or "Promozione esclusiva"
    )
    feed_bytes, resolved_label = await generate_campaign_graphic(concept_source, tenant)
    if not feed_bytes:
        raise HTTPException(status_code=500, detail="Generazione grafica fallita, riprova")

    image_version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    image_path = f"{tenant['id']}/campaigns/{campaign_id}/cover-{image_version}.jpg"
    try:
        sb.storage.from_("social-media").upload(
            path=image_path,
            file=io.BytesIO(feed_bytes),
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
    except Exception:
        sb.storage.from_("social-media").upload(
            path=image_path,
            file=feed_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )

    image_url = sb.storage.from_("social-media").get_public_url(image_path)

    target_summary["image_url"] = image_url
    if resolved_label:
        target_summary["treatment_label"] = resolved_label
    sb.table("wa_campaigns").update({"target_summary": target_summary}).eq("id", campaign_id).execute()

    center_name = tenant.get("display_name") or tenant.get("name") or "Centro Estetico"
    return {
        "image_url": image_url,
        "treatment_label": resolved_label or treatment_label,
        "center_name": center_name,
    }


@router.get("/{campaign_id}")
async def get_campaign(campaign_id: str, tenant: dict = Depends(get_tenant)):
    sb = get_supabase()
    res = (
        sb.table("wa_campaigns")
        .select("*")
        .eq("id", campaign_id)
        .eq("tenant_id", tenant["id"])
        .limit(1)
        .execute()
    )
    data = res.data or []
    if not data:
        raise HTTPException(status_code=404, detail="Campagna non trovata")
    return data[0]
