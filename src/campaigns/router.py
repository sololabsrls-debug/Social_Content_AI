import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from src.campaigns.agent import run_campaign_agent
from src.campaigns.models import CampaignChatRequest
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
