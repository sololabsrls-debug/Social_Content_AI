"""Strutture dati interne del modulo social."""

from typing import TypedDict, Optional


class ChecklistItem(TypedDict):
    id: str
    label: str           # "Foto PRIMA"
    instructions: str    # istruzioni in linguaggio semplice
    required: bool
    uploaded_url: Optional[str]


class AppointmentData(TypedDict):
    id: str
    start_at: str
    service_name: str
    service_id: Optional[str]
    service_category: Optional[str]
    descrizione_breve: Optional[str]
    benefici: Optional[list]
    prodotti_utilizzati: Optional[list]
    staff_name: Optional[str]
    notes: Optional[str]


class TenantBrandProfile(TypedDict):
    tenant_id: str
    name: str
    display_name: Optional[str]
    bio: Optional[str]
    logo_url: Optional[str]
    theme_primary_color: Optional[str]
    theme_secondary_color: Optional[str]
    social_profile: Optional[dict]   # tone_of_voice, style, brand_keywords, content_frequency...


class ContentPlan(TypedDict):
    """Piano per un singolo contenuto — output della selezione Gemini."""
    appointment_id: Optional[str]
    service_name: str
    archetype: str           # before_after | editorial | promo | educational | ...
    content_type: str        # post | story | reel | carousel
    scheduled_day: str       # lunedi | martedi | ...
    rationale: str
    material_checklist: list[ChecklistItem]
    caption_text: str
    hashtags: list[str]


class ContentRecord(TypedDict):
    """Record completo di social_content dal DB."""
    id: str
    tenant_id: str
    appointment_id: Optional[str]
    service_id: Optional[str]
    week_start: str
    week_end: str
    scheduled_date: Optional[str]
    platform: str
    content_type: str
    archetype: str
    material_checklist: list
    photos_input: list[str]
    client_consent: str
    estetista_notes: Optional[str]
    visual_brief: Optional[str]
    visual_brief_override: Optional[str]
    caption_text: Optional[str]
    hashtags: list[str]
    image_url_feed: Optional[str]
    image_url_story: Optional[str]
    status: str
