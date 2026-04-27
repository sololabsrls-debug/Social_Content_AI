import io
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from src.campaigns.agent import run_campaign_agent
from src.campaigns.models import CampaignChatRequest
from src.campaigns.wa_sender import send_whatsapp_message
from src.social.gemini_social import generate_image
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
    if not campaign_id:
        sb = get_supabase()
        res = sb.table("wa_campaigns").insert({
            "tenant_id": tenant_id,
            "status": "analyzing",
            "chat_history": [m.model_dump() for m in req.messages],
        }).execute()
        if not res.data:
            logger.error("Failed to create campaign record for tenant %s", tenant_id)
            raise HTTPException(status_code=500, detail="Impossibile creare la campagna")
        campaign_id = res.data[0]["id"]

    messages = [m.model_dump() for m in req.messages]

    async def stream():
        try:
            async for event_type, data in run_campaign_agent(messages, tenant_id, campaign_id):
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
    client_phones: list[str] = target_summary.get("client_phones") or []
    if not client_phones:
        raise HTTPException(status_code=400, detail="Nessun destinatario trovato — rianalizza la campagna")

    sent = 0
    failed = 0
    for phone in client_phones:
        result = await send_whatsapp_message(phone, message_text, tenant["id"])
        if result.get("ok"):
            sent += 1
        else:
            failed += 1
            logger.warning("WA send failed to %s: %s", phone, result.get("error"))

    sb.table("wa_campaigns").update({
        "status": "sent",
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "recipients_count": sent,
    }).eq("id", campaign_id).execute()

    return {"sent": sent, "failed": failed, "total": len(client_phones)}


@router.post("/{campaign_id}/generate-image")
async def generate_campaign_image(campaign_id: str, tenant: dict = Depends(get_tenant)):
    """Generate a social graphic for the campaign using the existing Gemini pipeline."""
    sb = get_supabase()
    res = (
        sb.table("wa_campaigns")
        .select("message_text, reason_text, target_summary")
        .eq("id", campaign_id)
        .eq("tenant_id", tenant["id"])
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Campagna non trovata")
    campaign = rows[0]

    message_text = campaign.get("message_text") or ""
    reason_text = campaign.get("reason_text") or ""
    if not message_text:
        raise HTTPException(status_code=400, detail="Messaggio campagna non impostato — genera prima la campagna")

    center_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    visual_brief = (
        f"Professional marketing graphic for a beauty center WhatsApp campaign.\n"
        f"Center: {center_name}\n"
        f"Campaign goal: {reason_text or 'Re-engagement promotion'}\n"
        f"Message theme: {message_text[:200]}\n"
        f"Style: elegant, feminine, accessible luxury. No people visible. "
        f"Focus on beauty products, spa atmosphere, or abstract elegance."
    )

    content_record = {
        "id": campaign_id,
        "photos_input": [],
        "archetype": "editorial",
        "visual_brief": visual_brief,
        "visual_brief_override": None,
        "client_consent": "no_client",
        "service": {},
        "service_name": "",
    }

    feed_bytes, _ = await generate_image(content_record, tenant)
    if not feed_bytes:
        raise HTTPException(status_code=500, detail="Generazione grafica fallita — riprova")

    image_path = f"{tenant['id']}/campaigns/{campaign_id}/cover.jpg"
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

    target_summary = campaign.get("target_summary") or {}
    target_summary["image_url"] = image_url
    sb.table("wa_campaigns").update({"target_summary": target_summary}).eq("id", campaign_id).execute()

    return {"image_url": image_url}


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
