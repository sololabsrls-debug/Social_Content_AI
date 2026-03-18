"""Endpoint REST del modulo social — chiamati da Lovable."""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel

from src.social.supabase_queries import (
    get_tenant_by_api_key,
    get_social_content_by_id,
    get_social_content_list,
    update_social_content,
    save_tenant_brand_profile,
)

logger = logging.getLogger("SOCIAL.router")

router = APIRouter(tags=["social"])


# ── Auth ───────────────────────────────────────────────────────────

async def get_tenant(x_api_key: str = Header(..., alias="X-API-Key")) -> dict:
    """Dependency: valida API key e ritorna il tenant."""
    tenant = get_tenant_by_api_key(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="API key non valida")
    return tenant


# ── Modelli request ────────────────────────────────────────────────

class GenerateWeekRequest(BaseModel):
    week_start: Optional[str] = None  # ISO date YYYY-MM-DD, default = prossima settimana
    tenant_id: Optional[str] = None   # ignorato (arriva dall'API key), accettato per compat


class UpdateContentRequest(BaseModel):
    photos_input: Optional[list[str]] = None
    client_consent: Optional[str] = None
    estetista_notes: Optional[str] = None
    caption_text: Optional[str] = None
    hashtags: Optional[list[str]] = None
    scheduled_date: Optional[str] = None


class UpdateBriefRequest(BaseModel):
    visual_brief_override: str


class UpdateStatusRequest(BaseModel):
    status: str


class BrandProfileRequest(BaseModel):
    # Identità
    tagline: Optional[str] = None              # "La bellezza autentica, ogni giorno"
    city: Optional[str] = None                 # "Milano"
    price_positioning: Optional[str] = None    # economico | mid-range | premium | luxury
    unique_selling_point: Optional[str] = None # "Uniche in città con laser XYZ"

    # Audience
    target_description: Optional[str] = None   # "Donne 30-50 anni, professioniste..."

    # Voce
    tone_of_voice: Optional[str] = None        # "caldo e professionale"
    communication_style: Optional[str] = None  # "informale, dai del tu"
    emoji_usage: Optional[str] = None          # "moderato (2-3)" | "pochi (1-2)" | "nessuno"
    avoid_words: Optional[list[str]] = None    # parole/frasi da non usare mai
    signature_phrases: Optional[list[str]] = None  # frasi ricorrenti del brand

    # Contenuto
    content_pillars: Optional[list[str]] = None    # temi editoriali
    brand_hashtags: Optional[list[str]] = None     # hashtag fissi sempre inclusi
    brand_keywords: Optional[list[str]] = None     # (legacy, mantenuto)
    content_frequency: Optional[int] = None
    platforms: Optional[list[str]] = None
    generation_day: Optional[str] = None
    generation_hour: Optional[int] = None

    # Visivo
    style: Optional[str] = None               # minimal | luxury | naturale | colorato


# ── Endpoints ──────────────────────────────────────────────────────

@router.post("/social/generate-week")
async def generate_week(
    req: GenerateWeekRequest,
    background_tasks: BackgroundTasks,
    tenant: dict = Depends(get_tenant),
):
    """
    Trigger manuale generazione piano settimanale.
    La generazione avviene in background (non blocca la risposta).
    """
    from src.social.content_pipeline import run_weekly_pipeline

    async def _run():
        await run_weekly_pipeline(tenant["id"], week_start_override=req.week_start)

    background_tasks.add_task(_run)
    return {"message": "Generazione piano avviata", "tenant_id": tenant["id"]}


@router.get("/social/content")
async def list_content(
    week_start: Optional[str] = None,
    week_end: Optional[str] = None,
    status: Optional[str] = None,
    tenant: dict = Depends(get_tenant),
):
    """Lista contenuti per il calendario Lovable."""
    contents = get_social_content_list(
        tenant_id=tenant["id"],
        week_start=week_start,
        week_end=week_end,
        status=status,
    )
    return {"data": contents, "count": len(contents)}


@router.get("/social/content/{content_id}")
async def get_content(
    content_id: str,
    tenant: dict = Depends(get_tenant),
):
    """Singolo contenuto con tutti i dettagli."""
    content = get_social_content_by_id(content_id)
    if not content or content["tenant_id"] != tenant["id"]:
        raise HTTPException(status_code=404, detail="Contenuto non trovato")
    return content


@router.patch("/social/content/{content_id}")
async def update_content(
    content_id: str,
    req: UpdateContentRequest,
    background_tasks: BackgroundTasks,
    tenant: dict = Depends(get_tenant),
):
    """
    Aggiorna un contenuto (foto, consenso, note, caption, hashtag).
    Se vengono caricate le foto, avvia automaticamente la generazione del brief.
    """
    content = get_social_content_by_id(content_id)
    if not content or content["tenant_id"] != tenant["id"]:
        raise HTTPException(status_code=404, detail="Contenuto non trovato")

    updates = {k: v for k, v in req.model_dump().items() if v is not None}

    if updates:
        update_social_content(content_id, updates)

    # Se sono state caricate foto → genera brief in background
    if req.photos_input and len(req.photos_input) > 0:
        from src.social.content_pipeline import generate_brief

        update_social_content(content_id, {
            "status": "material_ready",
            "photos_input": req.photos_input,
        })

        async def _brief():
            await generate_brief(content_id)

        background_tasks.add_task(_brief)
        return {"message": "Materiale ricevuto, generazione brief avviata", "content_id": content_id}

    return {"message": "Contenuto aggiornato", "content_id": content_id}


@router.patch("/social/content/{content_id}/brief")
async def update_brief(
    content_id: str,
    req: UpdateBriefRequest,
    tenant: dict = Depends(get_tenant),
):
    """Salva override del brief visivo scritto dall'estetista."""
    content = get_social_content_by_id(content_id)
    if not content or content["tenant_id"] != tenant["id"]:
        raise HTTPException(status_code=404, detail="Contenuto non trovato")

    update_social_content(content_id, {"visual_brief_override": req.visual_brief_override})
    return {"message": "Brief aggiornato", "content_id": content_id}


@router.post("/social/content/{content_id}/generate-image")
async def generate_image_endpoint(
    content_id: str,
    background_tasks: BackgroundTasks,
    tenant: dict = Depends(get_tenant),
):
    """
    Avvia generazione immagine dopo approvazione brief.
    Eseguita in background — Lovable riceve aggiornamento via Supabase Realtime.
    """
    content = get_social_content_by_id(content_id)
    if not content or content["tenant_id"] != tenant["id"]:
        raise HTTPException(status_code=404, detail="Contenuto non trovato")

    if content["status"] not in ("brief_ready", "draft"):
        raise HTTPException(
            status_code=400,
            detail=f"Status '{content['status']}' non valido per generazione immagine"
        )

    from src.social.content_pipeline import generate_image_for_content

    async def _gen():
        await generate_image_for_content(content_id)

    background_tasks.add_task(_gen)
    return {"message": "Generazione immagine avviata", "content_id": content_id}


@router.patch("/social/content/{content_id}/status")
async def update_status(
    content_id: str,
    req: UpdateStatusRequest,
    tenant: dict = Depends(get_tenant),
):
    """Approva, rifiuta o aggiorna status di un contenuto."""
    VALID_STATUSES = {"approved", "rejected", "published"}
    if req.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Status non valido. Valori accettati: {VALID_STATUSES}"
        )

    content = get_social_content_by_id(content_id)
    if not content or content["tenant_id"] != tenant["id"]:
        raise HTTPException(status_code=404, detail="Contenuto non trovato")

    from datetime import datetime
    import pytz
    now = datetime.now(pytz.timezone("Europe/Rome")).isoformat()

    extra = {}
    if req.status == "approved":
        extra["approved_at"] = now
    elif req.status == "published":
        extra["published_at"] = now

    update_social_content(content_id, {"status": req.status, **extra})
    return {"message": f"Status aggiornato a '{req.status}'", "content_id": content_id}


@router.post("/social/brand-profile")
async def save_brand_profile(
    req: BrandProfileRequest,
    tenant: dict = Depends(get_tenant),
):
    """Salva/aggiorna profilo brand del tenant."""
    current = tenant.get("social_profile") or {}
    updated = {**current, **{k: v for k, v in req.model_dump().items() if v is not None}}
    save_tenant_brand_profile(tenant["id"], updated)
    return {"message": "Profilo brand salvato", "social_profile": updated}


@router.get("/social/health")
async def health():
    return {"status": "ok", "service": "social-content-ai", "version": "5.5-clean-delicate"}
