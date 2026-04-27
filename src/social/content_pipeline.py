"""
Orchestratore del modulo social.

Tre entry point principali:
- run_weekly_pipeline(tenant_id)     → chiamato domenica dallo scheduler
- generate_brief(content_id)         → chiamato quando estetista carica materiale
- generate_image_for_content(content_id) → chiamato quando estetista approva il brief
"""

import io
import logging
from datetime import date, timedelta, datetime

import pytz

from src.supabase_client import get_supabase
from src.social.supabase_queries import (
    get_active_tenants,
    get_week_appointments,
    get_existing_week_content,
    create_social_content,
    get_social_content_by_id,
    update_social_content,
    get_appointment_with_service,
    get_manual_content_count,
)
from src.social.gemini_social import (
    select_and_plan_week,
    generate_visual_brief,
    generate_image,
    generate_image_variants,
    generate_ai_graphic_post,
)

logger = logging.getLogger("SOCIAL.pipeline")

ROME_TZ = pytz.timezone("Europe/Rome")

# Mappa giorno testuale → offset dalla settimana
DAY_OFFSETS = {
    "lunedi": 0, "martedi": 1, "mercoledi": 2,
    "giovedi": 3, "venerdi": 4, "sabato": 5, "domenica": 6,
}


def _next_week_bounds() -> tuple[date, date]:
    """Lunedì e domenica della settimana prossima (timezone Roma)."""
    today = datetime.now(ROME_TZ).date()
    days_to_monday = (7 - today.weekday()) % 7 or 7
    next_monday = today + timedelta(days=days_to_monday)
    next_sunday = next_monday + timedelta(days=6)
    return next_monday, next_sunday


def _save_images_to_storage(
    tenant_id: str,
    content_id: str,
    feed_bytes: bytes,
    story_bytes: bytes,
    suffix: str = "",
) -> tuple[str, str]:
    """Salva feed e story su Supabase Storage, ritorna le URL pubbliche."""
    sb = get_supabase()
    bucket = "social-media"

    feed_path = f"{tenant_id}/{content_id}/feed{suffix}.jpg"
    story_path = f"{tenant_id}/{content_id}/story{suffix}.jpg"

    try:
        sb.storage.from_(bucket).upload(
            path=feed_path,
            file=io.BytesIO(feed_bytes),
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
    except Exception as e:
        logger.warning(f"Upload feed fallito, provo con bytes: {e}")
        sb.storage.from_(bucket).upload(
            path=feed_path,
            file=feed_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )

    try:
        sb.storage.from_(bucket).upload(
            path=story_path,
            file=io.BytesIO(story_bytes),
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
    except Exception as e:
        logger.warning(f"Upload story fallito, provo con bytes: {e}")
        sb.storage.from_(bucket).upload(
            path=story_path,
            file=story_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )

    feed_url = sb.storage.from_(bucket).get_public_url(feed_path)
    story_url = sb.storage.from_(bucket).get_public_url(story_path)
    return feed_url, story_url


# ── 1. Pipeline settimanale ────────────────────────────────────────

async def run_weekly_pipeline(tenant_id: str, week_start_override: str | None = None) -> dict:
    """
    Pipeline completa per un tenant:
    1. Calcola settimana (usa week_start_override se fornito, altrimenti prossima settimana)
    2. Controlla idempotenza (non rigenera se già pianificato)
    3. Legge appuntamenti
    4. Gemini seleziona e crea piano con istruzioni
    5. Salva record social_content (status: waiting_material)

    Ritorna summary.
    """
    if week_start_override:
        # Usa la settimana specificata dal frontend
        from datetime import date as date_type
        week_start = date_type.fromisoformat(week_start_override)
        week_end = week_start + timedelta(days=6)
    else:
        week_start, week_end = _next_week_bounds()

    # Idempotenza: se esiste già contenuto per questa settimana, salta
    existing = get_existing_week_content(tenant_id, week_start)
    if existing:
        logger.info(
            f"Tenant {tenant_id}: piano settimana {week_start} già esistente "
            f"({len(existing)} contenuti), skip"
        )
        return {"tenant_id": tenant_id, "week_start": str(week_start), "skipped": True}

    # Leggi appuntamenti
    appointments = get_week_appointments(tenant_id, week_start, week_end)
    if not appointments:
        logger.info(f"Tenant {tenant_id}: nessun appuntamento per la settimana {week_start}")
        return {"tenant_id": tenant_id, "week_start": str(week_start), "no_appointments": True}

    # Leggi profilo tenant
    sb = get_supabase()
    tenant_res = (
        sb.table("tenants")
        .select("id, name, display_name, bio, logo_url, theme_primary_color, "
                "theme_secondary_color, social_profile")
        .eq("id", tenant_id)
        .maybe_single()
        .execute()
    )
    tenant = tenant_res.data
    if not tenant:
        logger.error(f"Tenant {tenant_id} non trovato")
        return {"tenant_id": tenant_id, "error": "tenant not found"}

    # Gemini seleziona e pianifica
    selected = await select_and_plan_week(
        appointments=appointments,
        tenant=tenant,
        week_start_str=str(week_start),
        week_end_str=str(week_end),
        existing_appointment_ids=[],
    )

    if not selected:
        logger.info(f"Tenant {tenant_id}: Gemini non ha selezionato contenuti")
        return {"tenant_id": tenant_id, "week_start": str(week_start), "records_created": 0}

    # Crea record per ogni contenuto selezionato
    records_created = 0
    social_profile = tenant.get("social_profile") or {}
    platforms = social_profile.get("platforms", ["instagram", "facebook"])
    platform_value = "both" if len(platforms) > 1 else (platforms[0] if platforms else "both")

    for plan in selected:
        appt_id = plan.get("appointment_id")
        service_id = None

        # Recupera service_id, service_name e data reale dall'appuntamento
        service_name_from_appt = plan.get("service_name")
        scheduled_date = week_start  # fallback
        if appt_id:
            for a in appointments:
                if a.get("id") == appt_id:
                    service = a.get("service") or {}
                    service_id = service.get("id")
                    if not service_name_from_appt:
                        service_name_from_appt = service.get("name")
                    # Usa la data reale dell'appuntamento, non quella inventata da Gemini
                    appt_date_str = (a.get("start_at") or "")[:10]
                    if appt_date_str:
                        from datetime import date as date_type
                        scheduled_date = date_type.fromisoformat(appt_date_str)
                    break

        record = {
            "tenant_id": tenant_id,
            "appointment_id": appt_id,
            "service_id": service_id,
            "week_start": str(week_start),
            "week_end": str(week_end),
            "scheduled_date": str(scheduled_date),
            "platform": platform_value,
            "content_type": plan.get("content_type", "post"),
            "archetype": plan.get("archetype", "editorial"),
            "material_checklist": plan.get("material_checklist", []),
            "caption_text": plan.get("caption_text"),
            "hashtags": plan.get("hashtags", []),
            "selection_rationale": plan.get("rationale"),
            "status": "waiting_material",
            "photos_input": [],
            "client_consent": "details_only",
        }

        try:
            create_social_content(record)
            records_created += 1
        except Exception as e:
            logger.error(f"Errore salvataggio record social_content: {e}")

    # ── Post AI Graphic settimanale ──────────────────────────────
    try:
        ai_post = await generate_ai_graphic_post(tenant)

        ai_feed_bytes = ai_post.pop("feed_bytes", None)
        ai_story_bytes = ai_post.pop("story_bytes", None)
        ai_category = ai_post.pop("ai_graphic_category", "tip_beauty")

        # Carica immagini su Storage se generate
        ai_feed_url, ai_story_url = None, None
        if ai_feed_bytes and ai_story_bytes:
            import uuid as _uuid
            tmp_id = str(_uuid.uuid4())
            ai_feed_url, ai_story_url = _save_images_to_storage(
                tenant_id=tenant_id,
                content_id=tmp_id,
                feed_bytes=ai_feed_bytes,
                story_bytes=ai_story_bytes,
                suffix="_ai",
            )

        # Giorno pubblicazione: sabato della settimana
        ai_scheduled_date = str(week_start + timedelta(days=5))

        ai_record = {
            "tenant_id": tenant_id,
            "appointment_id": None,
            "service_id": ai_post.get("service_id"),
            "week_start": str(week_start),
            "week_end": str(week_end),
            "scheduled_date": ai_scheduled_date,
            "platform": platform_value,
            "content_type": ai_post.get("content_type", "post"),
            "archetype": "ai_graphic",
            "material_checklist": [],
            "caption_text": ai_post.get("caption_text"),
            "hashtags": ai_post.get("hashtags", []),
            "selection_rationale": ai_post.get("selection_rationale"),
            "status": "draft",
            "photos_input": [],
            "client_consent": "no_client",
            "image_url_feed": ai_feed_url,
            "image_url_story": ai_story_url,
            "image_generated_at": datetime.now(ROME_TZ).isoformat(),
        }

        create_social_content(ai_record)
        records_created += 1
        logger.info(f"[ai_graphic] post creato — categoria: {ai_category}")

        # Aggiorna rotazione nel social_profile del tenant
        updated_profile = {**social_profile, "last_ai_graphic_category": ai_category}
        sb.table("tenants").update({"social_profile": updated_profile}).eq("id", tenant_id).execute()

    except Exception as e:
        logger.error(f"[ai_graphic] errore generazione post settimanale: {e}")
        # Non blocca il resto del pipeline

    logger.info(
        f"Tenant {tenant_id}: piano settimana {week_start} creato "
        f"({records_created} contenuti)"
    )
    return {
        "tenant_id": tenant_id,
        "week_start": str(week_start),
        "records_created": records_created,
    }


async def create_manual_content(
    tenant_id: str,
    appointment_id: str,
    week_start_str: str,
) -> dict:
    """
    Crea un singolo contenuto manuale per un appuntamento scelto dall'estetista.
    Usa la stessa logica di rotazione archetype della pipeline settimanale.
    Max 1 contenuto manuale per settimana per tenant.
    """
    from datetime import date as date_type
    from src.social.gemini_social import _get_rules_with_rotation

    week_start = date_type.fromisoformat(week_start_str)
    week_end = week_start + timedelta(days=6)

    count = get_manual_content_count(tenant_id, week_start)
    if count >= 1:
        raise ValueError("Hai già aggiunto un contenuto manuale questa settimana")

    appt = get_appointment_with_service(appointment_id, tenant_id)
    if not appt:
        raise ValueError("Appuntamento non trovato")

    service = appt.get("service") or {}
    service_id = service.get("id")
    service_name = service.get("name", "Trattamento")

    scheduled_date_str = (appt.get("start_at") or "")[:10]
    scheduled_date = (
        date_type.fromisoformat(scheduled_date_str)
        if scheduled_date_str
        else week_start
    )

    sb = get_supabase()
    tenant_res = (
        sb.table("tenants")
        .select("id, name, display_name, social_profile")
        .eq("id", tenant_id)
        .maybe_single()
        .execute()
    )
    tenant = tenant_res.data or {}
    social_profile = tenant.get("social_profile") or {}
    platforms = social_profile.get("platforms", ["instagram", "facebook"])
    platform_value = "both" if len(platforms) > 1 else (platforms[0] if platforms else "both")

    rules = await _get_rules_with_rotation(service_name, service_id, tenant_id)

    record = {
        "tenant_id": tenant_id,
        "appointment_id": appointment_id,
        "service_id": service_id,
        "week_start": str(week_start),
        "week_end": str(week_end),
        "scheduled_date": str(scheduled_date),
        "platform": platform_value,
        "content_type": "post",
        "archetype": rules["archetype"],
        "material_checklist": rules["checklist"],
        "status": "waiting_material",
        "is_manual": True,
        "photos_input": [],
        "client_consent": "details_only",
    }

    created = create_social_content(record)
    logger.info(
        f"Contenuto manuale creato: tenant={tenant_id} appt={appointment_id} "
        f"archetype={rules['archetype']} content_id={created.get('id')}"
    )
    return created


async def run_all_tenants() -> list[dict]:
    """Esegue la pipeline per tutti i tenant attivi."""
    tenants = get_active_tenants()
    results = []
    for tenant in tenants:
        try:
            result = await run_weekly_pipeline(tenant["id"])
            results.append(result)
        except Exception as e:
            logger.error(f"Errore pipeline tenant {tenant['id']}: {e}")
            results.append({"tenant_id": tenant["id"], "error": str(e)})
    return results


# ── 2. Genera brief visivo ────────────────────────────────────────

async def generate_brief(content_id: str) -> bool:
    """
    Chiamato quando l'estetista ha caricato il materiale.
    Analizza le foto con Gemini e genera il brief visivo testuale.
    Aggiorna status → brief_ready.
    """
    content = get_social_content_by_id(content_id)
    if not content:
        logger.error(f"Content {content_id} non trovato")
        return False

    tenant_id = content["tenant_id"]
    sb = get_supabase()
    tenant_res = (
        sb.table("tenants")
        .select("id, name, display_name, bio, logo_url, theme_primary_color, "
                "theme_secondary_color, social_profile")
        .eq("id", tenant_id)
        .maybe_single()
        .execute()
    )
    tenant = tenant_res.data or {}

    try:
        brief = await generate_visual_brief(content, tenant)
        update_social_content(content_id, {
            "visual_brief": brief,
            "status": "brief_ready",
            "brief_generated_at": datetime.now(ROME_TZ).isoformat(),
        })
        logger.info(f"Brief generato per content {content_id}")
        return True
    except Exception as e:
        logger.error(f"Errore generazione brief {content_id}: {e}")
        return False


# ── 3. Genera immagine finale ────────────────────────────────────

async def generate_image_for_content(content_id: str) -> bool:
    """
    Chiamato quando l'estetista approva il brief.
    Genera 3 varianti in parallelo con direzioni creative diverse.
    Salva su Supabase Storage e aggiorna status → variants_ready.
    """
    update_social_content(content_id, {"status": "generating"})

    content = get_social_content_by_id(content_id)
    if not content:
        logger.error(f"Content {content_id} non trovato")
        return False

    tenant_id = content["tenant_id"]
    sb = get_supabase()
    tenant_res = (
        sb.table("tenants")
        .select("id, name, display_name, bio, logo_url, theme_primary_color, "
                "theme_secondary_color, social_profile")
        .eq("id", tenant_id)
        .maybe_single()
        .execute()
    )
    tenant = tenant_res.data or {}

    try:
        variants = await generate_image_variants(content, tenant)

        if not variants:
            update_social_content(content_id, {"status": "brief_ready"})
            return False

        # Salva ogni variante su Supabase Storage
        saved_variants = []
        for v in variants:
            suffix = f"_v{v['index']}"
            feed_url, story_url = _save_images_to_storage(
                tenant_id, content_id,
                v["feed_bytes"], v.get("story_bytes") or b"",
                suffix=suffix,
            )
            saved_variants.append({
                "index": v["index"],
                "direction": v["direction"],
                "feed_url": feed_url,
                "story_url": story_url,
            })

        update_social_content(content_id, {
            "image_variants": saved_variants,
            "status": "variants_ready",
            "image_generated_at": datetime.now(ROME_TZ).isoformat(),
        })

        logger.info(f"{len(saved_variants)} varianti generate per content {content_id}")
        return True

    except Exception as e:
        logger.error(f"Errore generazione varianti {content_id}: {e}")
        update_social_content(content_id, {"status": "brief_ready"})
        return False
