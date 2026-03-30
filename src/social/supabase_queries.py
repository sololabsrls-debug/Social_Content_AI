"""Tutte le query Supabase del modulo social."""

import logging
from datetime import date, timedelta
from typing import Optional

from src.supabase_client import get_supabase

logger = logging.getLogger("SOCIAL.db")


def get_active_tenants() -> list[dict]:
    """Tutti i tenant attivi con social_profile configurato."""
    sb = get_supabase()
    res = (
        sb.table("tenants")
        .select("id, name, display_name, bio, logo_url, theme_primary_color, "
                "theme_secondary_color, social_profile, content_api_key")
        .eq("status", "active")
        .execute()
    )
    return res.data or []


def get_tenant_by_api_key(api_key: str) -> Optional[dict]:
    """Cerca tenant per content_api_key — usato per auth endpoint."""
    sb = get_supabase()
    res = (
        sb.table("tenants")
        .select("id, name, display_name, bio, logo_url, theme_primary_color, "
                "theme_secondary_color, social_profile, content_api_key")
        .eq("content_api_key", api_key)
        .eq("status", "active")
        .maybe_single()
        .execute()
    )
    return res.data


def get_week_appointments(tenant_id: str, week_start: date, week_end: date) -> list[dict]:
    """Appuntamenti confermati/pending per la settimana con dati servizio e staff."""
    sb = get_supabase()
    res = (
        sb.table("appointments")
        .select(
            "id, start_at, end_at, notes, status, "
            "service:services(id, name, category_id, descrizione_breve, "
            "descrizione_completa, benefici, prodotti_utilizzati, duration_min), "
            "staff:staff(name)"
        )
        .eq("tenant_id", tenant_id)
        .in_("status", ["confirmed", "pending"])
        .gte("start_at", week_start.isoformat())
        .lte("start_at", week_end.isoformat())
        .order("start_at")
        .execute()
    )
    return res.data or []


def get_existing_week_content(tenant_id: str, week_start: date) -> list[dict]:
    """Contenuti già pianificati per questa settimana — per idempotenza."""
    sb = get_supabase()
    res = (
        sb.table("social_content")
        .select("id, appointment_id, archetype, status")
        .eq("tenant_id", tenant_id)
        .eq("week_start", week_start.isoformat())
        .execute()
    )
    return res.data or []


def create_social_content(data: dict) -> dict:
    """Inserisce un nuovo record social_content."""
    sb = get_supabase()
    res = sb.table("social_content").insert(data).execute()
    return res.data[0] if res.data else {}


def get_social_content_by_id(content_id: str) -> Optional[dict]:
    """Singolo record con join a services e appointments."""
    sb = get_supabase()
    res = (
        sb.table("social_content")
        .select(
            "*, "
            "service:services(name, descrizione_breve, benefici, prodotti_utilizzati), "
            "appointment:appointments(start_at, notes)"
        )
        .eq("id", content_id)
        .maybe_single()
        .execute()
    )
    return res.data


def get_social_content_list(
    tenant_id: str,
    week_start: Optional[str] = None,
    week_end: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict]:
    """Lista contenuti per il calendario Lovable."""
    sb = get_supabase()
    q = (
        sb.table("social_content")
        .select(
            "id, tenant_id, appointment_id, service_id, week_start, week_end, "
            "scheduled_date, platform, content_type, archetype, status, "
            "material_checklist, photos_input, client_consent, "
            "visual_brief, caption_text, hashtags, image_url_feed, image_url_story, "
            "created_at, updated_at, "
            "service:services(name)"
        )
        .eq("tenant_id", tenant_id)
    )
    if week_start:
        q = q.gte("week_start", week_start)
    if week_end:
        q = q.lte("week_end", week_end)
    if status:
        q = q.eq("status", status)
    res = q.order("scheduled_date", desc=False).execute()
    return res.data or []


def update_social_content(content_id: str, data: dict) -> dict:
    """Aggiorna campi di un record social_content."""
    sb = get_supabase()
    res = (
        sb.table("social_content")
        .update(data)
        .eq("id", content_id)
        .execute()
    )
    return res.data[0] if res.data else {}


def save_tenant_brand_profile(tenant_id: str, social_profile: dict) -> None:
    """Salva il profilo brand nel campo social_profile del tenant."""
    sb = get_supabase()
    sb.table("tenants").update({"social_profile": social_profile}).eq("id", tenant_id).execute()
