"""
Logica AI del modulo social — tutte le chiamate Gemini.

Tre funzioni principali:
1. select_and_plan_week()   → sceglie appuntamenti; Python decide archetype+checklist,
                              Gemini fa solo caption+hashtag
2. generate_visual_brief()  → analizza foto e descrive cosa creerà (PRIMA di generare)
3. generate_image()         → genera la grafica finale con gemini-3-pro-image-preview
"""

import asyncio
import io
import json
import logging
import os
import re
from typing import Optional

import httpx
from google import genai
from google.genai import types
from PIL import Image

logger = logging.getLogger("SOCIAL.gemini")

MODEL_TEXT = "gemini-2.5-flash"
MODEL_IMAGE = "gemini-3-pro-image-preview"

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


# ── Helpers ────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict | list:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    return json.loads(text)


def _center_crop_square(img: Image.Image) -> Image.Image:
    """Ritaglia l'immagine al centro per ottenere un quadrato 1:1."""
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


async def _download_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return _center_crop_square(img)


def _image_to_pil(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _create_story_version(feed_image: Image.Image) -> bytes:
    w, h = feed_image.size
    story_h = int(w * 16 / 9)
    background_color = (253, 245, 240)
    story = Image.new("RGB", (w, story_h), background_color)
    y_offset = (story_h - h) // 2
    story.paste(feed_image, (0, y_offset))
    out = io.BytesIO()
    story.save(out, format="JPEG", quality=90)
    return out.getvalue()


def _extract_image_from_response(response) -> Optional[bytes]:
    for part in response.candidates[0].content.parts:
        if part.inline_data and "image" in part.inline_data.mime_type:
            return part.inline_data.data
    return None


# ── Knowledge base V2: rotazione archetype per servizio ────────────
#
# Per ogni categoria di servizio:
#   - keywords:          per il matching
#   - rotation_weights:  target % per archetype (sum=100)
#   - archetypes:        checklist base per ogni archetype possibile
#
# Il sistema di rotazione controlla gli ultimi N post per quel servizio
# e sceglie l'archetype più "in debito" rispetto ai pesi target.
# Gemini personalizza poi le istruzioni foto con _personalize_checklist_instructions().

_SERVICE_RULES_V2: dict = {
    "ciglia": {
        "keywords": ["laminazione ciglia", "extension ciglia", "lifting ciglia", "ciglia"],
        "rotation_weights": {"before_after": 50, "educational": 25, "behind_scenes": 15, "editorial": 10},
        "archetypes": {
            "before_after": [
                {"id": "before", "label": "Foto ciglia PRIMA", "required": True, "instructions": ""},
                {"id": "after",  "label": "Foto ciglia DOPO",  "required": True, "instructions": ""},
            ],
            "educational": [
                {"id": "treatment", "label": "Foto durante il trattamento ciglia", "required": True,  "instructions": ""},
                {"id": "tools",     "label": "Kit prodotti usati",                 "required": False, "instructions": ""},
            ],
            "behind_scenes": [
                {"id": "process", "label": "Mani al lavoro sulle ciglia", "required": True, "instructions": ""},
            ],
            "editorial": [
                {"id": "closeup", "label": "Close-up risultato ciglia", "required": True, "instructions": ""},
            ],
        },
    },
    "sopracciglia": {
        "keywords": ["microblading", "nanoblading", "laminazione sopracciglia", "tinta sopracciglia", "sopracciglia"],
        "rotation_weights": {"before_after": 50, "editorial": 25, "educational": 15, "behind_scenes": 10},
        "archetypes": {
            "before_after": [
                {"id": "before", "label": "Foto sopracciglia PRIMA", "required": True, "instructions": ""},
                {"id": "after",  "label": "Foto sopracciglia DOPO",  "required": True, "instructions": ""},
            ],
            "editorial": [
                {"id": "closeup", "label": "Close-up sopracciglia finali", "required": True, "instructions": ""},
            ],
            "educational": [
                {"id": "treatment", "label": "Foto durante il disegno/trattamento", "required": True, "instructions": ""},
            ],
            "behind_scenes": [
                {"id": "process", "label": "Mani al lavoro sulle sopracciglia", "required": True, "instructions": ""},
            ],
        },
    },
    "unghie": {
        "keywords": ["semipermanente", "smalto", "nail art", "ricostruzione unghie", "manicure", "pedicure", "unghie"],
        "rotation_weights": {"editorial": 40, "behind_scenes": 25, "educational": 20, "before_after": 15},
        "archetypes": {
            "editorial": [
                {"id": "result", "label": "Foto risultato unghie", "required": True, "instructions": ""},
            ],
            "behind_scenes": [
                {"id": "process", "label": "Foto durante la manicure",         "required": True,  "instructions": ""},
                {"id": "detail",  "label": "Dettaglio strumenti o smalti usati", "required": False, "instructions": ""},
            ],
            "educational": [
                {"id": "technique", "label": "Foto che mostra la tecnica", "required": True, "instructions": ""},
            ],
            "before_after": [
                {"id": "before", "label": "Foto unghie PRIMA", "required": True, "instructions": ""},
                {"id": "after",  "label": "Foto unghie DOPO",  "required": True, "instructions": ""},
            ],
        },
    },
    "laser": {
        "keywords": ["laser", "luce pulsata", "epilazione laser"],
        "rotation_weights": {"educational": 45, "behind_scenes": 30, "editorial": 25},
        "archetypes": {
            "educational": [
                {"id": "machine",   "label": "Foto del macchinario laser",           "required": True,  "instructions": ""},
                {"id": "treatment", "label": "Trattamento in corso (zona neutrale)",  "required": False, "instructions": ""},
            ],
            "behind_scenes": [
                {"id": "treatment", "label": "Foto durante il trattamento", "required": True, "instructions": ""},
            ],
            "editorial": [
                {"id": "result", "label": "Foto della pelle dopo il trattamento", "required": True, "instructions": ""},
            ],
        },
    },
    "viso": {
        "keywords": ["pulizia viso", "idratazione viso", "trattamento viso", "peeling", "viso"],
        "rotation_weights": {"before_after": 40, "behind_scenes": 30, "educational": 20, "editorial": 10},
        "archetypes": {
            "before_after": [
                {"id": "before", "label": "Foto viso PRIMA", "required": True, "instructions": ""},
                {"id": "after",  "label": "Foto viso DOPO",  "required": True, "instructions": ""},
            ],
            "behind_scenes": [
                {"id": "process", "label": "Foto durante il trattamento viso", "required": True, "instructions": ""},
            ],
            "educational": [
                {"id": "product", "label": "Foto prodotti usati o step del trattamento", "required": True, "instructions": ""},
            ],
            "editorial": [
                {"id": "result", "label": "Foto viso dopo il trattamento", "required": True, "instructions": ""},
            ],
        },
    },
    "massaggio": {
        "keywords": ["massaggio", "drenante", "pressoterapia", "cavitazione", "radiofrequenza", "mesoterapia"],
        "rotation_weights": {"behind_scenes": 45, "educational": 30, "editorial": 25},
        "archetypes": {
            "behind_scenes": [
                {"id": "treatment", "label": "Foto durante il massaggio/trattamento", "required": True, "instructions": ""},
            ],
            "educational": [
                {"id": "machine", "label": "Macchinario o mani al lavoro", "required": True, "instructions": ""},
            ],
            "editorial": [
                {"id": "result", "label": "Zona trattata — risultato visibile", "required": True, "instructions": ""},
            ],
        },
    },
    "ceretta": {
        "keywords": ["ceretta", "cera", "epilazione cera", "wax", "sugaring"],
        "rotation_weights": {"before_after": 40, "editorial": 30, "behind_scenes": 20, "educational": 10},
        "archetypes": {
            "before_after": [
                {"id": "before", "label": "Foto zona PRIMA della ceretta", "required": True, "instructions": ""},
                {"id": "after",  "label": "Foto zona DOPO la ceretta", "required": True, "instructions": ""},
            ],
            "editorial": [
                {"id": "result", "label": "Foto della pelle dopo la ceretta (liscia)", "required": True, "instructions": ""},
            ],
            "behind_scenes": [
                {"id": "process", "label": "Foto durante l'applicazione della cera", "required": True, "instructions": ""},
            ],
            "educational": [
                {"id": "product", "label": "Foto dei prodotti/cera usati", "required": True, "instructions": ""},
            ],
        },
    },
    "epilazione": {
        "keywords": ["epilazione", "diodo", "alexandrite", "epilazione gambe", "epilazione braccia",
                     "epilazione ascelle", "epilazione viso", "epilazione corpo"],
        "rotation_weights": {"before_after": 35, "educational": 30, "behind_scenes": 20, "editorial": 15},
        "archetypes": {
            "before_after": [
                {"id": "before", "label": "Foto zona PRIMA dell'epilazione", "required": True, "instructions": ""},
                {"id": "after",  "label": "Foto zona DOPO l'epilazione", "required": True, "instructions": ""},
            ],
            "educational": [
                {"id": "machine", "label": "Foto del macchinario per epilazione", "required": True, "instructions": ""},
                {"id": "treatment", "label": "Foto del trattamento (zona neutrale)", "required": False, "instructions": ""},
            ],
            "behind_scenes": [
                {"id": "process", "label": "Foto durante il trattamento", "required": True, "instructions": ""},
            ],
            "editorial": [
                {"id": "result", "label": "Foto della pelle dopo l'epilazione", "required": True, "instructions": ""},
            ],
        },
    },
}

# Fallback per servizi non riconosciuti
_DEFAULT_RULES_V2: dict = {
    "rotation_weights": {"before_after": 25, "editorial": 35, "behind_scenes": 25, "educational": 15},
    "archetypes": {
        "before_after": [
            {"id": "before", "label": "Foto PRIMA del trattamento", "required": True, "instructions": ""},
            {"id": "after",  "label": "Foto DOPO il trattamento",  "required": True, "instructions": ""},
        ],
        "editorial": [
            {"id": "result", "label": "Foto del risultato", "required": True, "instructions": ""},
        ],
        "behind_scenes": [
            {"id": "process", "label": "Foto durante il trattamento", "required": True, "instructions": ""},
        ],
        "educational": [
            {"id": "treatment", "label": "Foto del trattamento", "required": True, "instructions": ""},
        ],
    },
}

# ── AI Graphic — rotazione categorie settimanali ──────────────────

AI_GRAPHIC_CATEGORIES = ["tip_beauty", "spotlight", "stagionale", "curiosita"]


def _get_next_ai_graphic_category(social_profile: dict) -> str:
    """
    Ritorna la categoria da usare questa settimana.
    Cicla tra AI_GRAPHIC_CATEGORIES basandosi su last_ai_graphic_category
    nel social_profile del tenant.
    """
    last = social_profile.get("last_ai_graphic_category")
    if not last or last not in AI_GRAPHIC_CATEGORIES:
        return AI_GRAPHIC_CATEGORIES[0]
    idx = AI_GRAPHIC_CATEGORIES.index(last)
    return AI_GRAPHIC_CATEGORIES[(idx + 1) % len(AI_GRAPHIC_CATEGORIES)]


async def _pick_spotlight_service(tenant_id: str) -> Optional[dict]:
    """
    Sceglie il servizio da mettere in spotlight questa settimana.
    Prioritizza servizi non recentemente spotlightati.
    Ritorna dict con id, name, descrizione_breve, benefici oppure None.
    """
    try:
        from src.supabase_client import get_supabase
        sb = get_supabase()

        # Tutti i servizi attivi del tenant
        services_res = (
            sb.table("services")
            .select("id, name, descrizione_breve, benefici")
            .eq("tenant_id", tenant_id)
            .eq("is_active", True)
            .execute()
        )
        services = services_res.data or []
        if not services:
            return None

        # Ultimi 8 spotlight (social_content con archetype=ai_graphic e service_id valorizzato)
        recent_res = (
            sb.table("social_content")
            .select("service_id")
            .eq("tenant_id", tenant_id)
            .eq("archetype", "ai_graphic")
            .not_.is_("service_id", "null")
            .order("created_at", desc=True)
            .limit(8)
            .execute()
        )
        recent_ids = {
            row["service_id"]
            for row in (recent_res.data or [])
            if row.get("service_id")
        }

        # Primo servizio non recente
        for s in services:
            if s["id"] not in recent_ids:
                return s

        # Se tutti recenti, ricomincia dal primo
        return services[0]

    except Exception as e:
        logger.warning(f"Errore scelta spotlight service per tenant {tenant_id}: {e}")
        return None


async def _generate_ai_graphic_text(
    category: str,
    tenant: dict,
    spotlight_service: Optional[dict] = None,
    system_prompt: str = "",
) -> tuple[str, str, list[str]]:
    """
    Genera concept, caption e hashtag per il post ai_graphic.
    Ritorna (concept, caption, hashtags).
    concept = tema/titolo principale del post (usato anche come guida per l'immagine).
    """
    sp = tenant.get("social_profile") or {}
    center_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    hashtags_per_post = sp.get("hashtags_per_post") or 10

    from datetime import datetime
    import pytz as _pytz
    month = datetime.now(_pytz.timezone("Europe/Rome")).month
    season_map = {
        12: "inverno", 1: "inverno", 2: "inverno",
        3: "primavera", 4: "primavera", 5: "primavera",
        6: "estate", 7: "estate", 8: "estate",
        9: "autunno", 10: "autunno", 11: "autunno",
    }
    season = season_map.get(month, "primavera")

    category_instructions = {
        "tip_beauty": (
            f'Crea un post "Tip Beauty" — un consiglio pratico e utile sulla cura della bellezza.\n'
            f"Il consiglio deve essere specifico e direttamente applicabile.\n"
            f"Scegli un tema coerente con i servizi di {center_name} (ciglia, sopracciglia, unghie, viso, corpo).\n"
            f"CONCEPT: il consiglio in max 15 parole (es. 'Come mantenere le ciglia laminate più a lungo').\n"
            f"CAPTION: sviluppa il consiglio in modo coinvolgente (3-5 frasi)."
        ),
        "spotlight": (
            f"Crea un post che mette in vetrina il servizio: "
            f"{spotlight_service.get('name', 'servizio') if spotlight_service else 'servizio del centro'}.\n"
            f"Descrizione: {spotlight_service.get('descrizione_breve', '') if spotlight_service else ''}\n"
            f"Benefici: {spotlight_service.get('benefici', '') if spotlight_service else ''}\n"
            f"CONCEPT: titolo del post (es. 'Scopri la Laminazione Ciglia — il segreto dello sguardo perfetto').\n"
            f"CAPTION: descrivi il servizio con entusiasmo, 2-3 benefici chiave, invita a prenotare."
        ),
        "stagionale": (
            f"Crea un post stagionale per {season} — mese {month}.\n"
            f"Collega il messaggio ai trattamenti consigliati per questa stagione.\n"
            f"CONCEPT: tema stagionale principale "
            f"(es. 'Primavera: il momento giusto per rinnovare la pelle').\n"
            f"CAPTION: collega il messaggio stagionale ai trattamenti del centro."
        ),
        "curiosita": (
            f"Crea un post 'Lo sapevi che...?' su un trattamento estetico offerto da {center_name}.\n"
            f"La curiosità deve essere sorprendente, vera e direttamente legata a un servizio del centro.\n"
            f"Esempi: durata reale di un trattamento, un beneficio inaspettato, un fatto scientifico sul trattamento.\n"
            f"CONCEPT: la curiosità in forma di domanda o affermazione (max 15 parole, es. 'Lo sapevi che la laminazione ciglia dura fino a 8 settimane?').\n"
            f"CAPTION: sviluppa la curiosità con 2-3 frasi di approfondimento + invita a scoprire il trattamento."
        ),
    }

    prompt = (
        f"Sei il social media manager di {center_name}, un centro estetico italiano.\n\n"
        f"{category_instructions.get(category, category_instructions['tip_beauty'])}\n\n"
        f"Rispondi SOLO con JSON:\n"
        f'{{\n'
        f'  "concept": "titolo/tema principale del post (max 20 parole)",\n'
        f'  "caption": "testo caption Instagram completo in italiano (3-5 frasi, CTA finale inclusa)",\n'
        f'  "hashtags": ["hashtag1", "hashtag2", ...]\n'
        f'}}\n\n'
        f"Regole caption:\n"
        f"- Tono: {sp.get('tone_of_voice', 'caldo e professionale')}\n"
        f"- {hashtags_per_post} hashtag rilevanti (mix brand + niche + trend italiani)\n"
        f"- NO hashtag nel corpo della caption, solo nell'array"
    )

    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=MODEL_TEXT,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=0.7,
                response_mime_type="application/json",
            ),
        )
        data = _parse_json_response(response.text)
        concept = data.get("concept", "")
        caption = data.get("caption", "")
        hashtags = data.get("hashtags", [])
        logger.debug(f"ai_graphic testo generato — concept: {concept!r}")
        return concept, caption, hashtags
    except Exception as e:
        logger.error(f"Errore generazione testo ai_graphic ({category}): {e}")
        center = tenant.get("display_name") or tenant.get("name", "il nostro centro")
        return (
            f"Post {category}",
            f"Scopri i nostri trattamenti presso {center}! Prenota ora → link in bio",
            [],
        )


async def _generate_ai_graphic_image(
    category: str,
    concept: str,
    tenant: dict,
    spotlight_service: Optional[dict] = None,
) -> tuple[Optional[bytes], Optional[bytes]]:
    """
    Genera la grafica Instagram da zero — nessuna foto reale in input.
    Ritorna (feed_bytes 1:1, story_bytes 9:16).
    """
    sp = tenant.get("social_profile") or {}
    center_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    primary_color = tenant.get("theme_primary_color") or "#6b2d4e"
    secondary_color = tenant.get("theme_secondary_color") or "#c9a0b4"
    accent_color = sp.get("accent_color") or secondary_color
    bg_color = sp.get("background_color") or "#fdf5f0"

    photo_style_labels = {
        "bright_natural": "bright and airy, natural light, vibrant but soft tones",
        "warm_moody": "warm and atmospheric, amber tones, soft shadows",
        "clean_white": "clean white background, flat professional lighting, minimal elegance",
        "dark_luxury": "dark luxury aesthetic, high contrast, premium feel",
    }
    visual_style = photo_style_labels.get(
        sp.get("photo_style") or "bright_natural",
        "bright and airy",
    )

    typo_labels = {
        "serif_elegant": "elegant serif typography (Playfair Display style)",
        "sans_modern": "clean modern sans-serif (Inter / Helvetica style)",
        "mixed": "mixed typography: serif headline, sans-serif body",
    }
    typo_style = typo_labels.get(
        sp.get("typography_style") or "serif_elegant",
        "elegant serif typography",
    )

    service_name = spotlight_service.get("name", "") if spotlight_service else ""
    service_benefits = spotlight_service.get("benefici", "") if spotlight_service else ""

    category_design = {
        "tip_beauty": (
            f"GRAPHIC TYPE: Beauty tip post\n"
            f"MAIN TEXT (Italian): {concept}\n\n"
            f"LAYOUT: Clean text-based graphic. Large readable tip text as the hero element. "
            f"Soft background using brand colors. Small decorative botanical or geometric element. "
            f"Center name '{center_name}' as subtle signature at the bottom."
        ),
        "spotlight": (
            f"GRAPHIC TYPE: Service spotlight promotional post\n"
            f"SERVICE: {service_name or concept}\n"
            f"KEY MESSAGE (Italian): {concept}\n"
            f"BENEFITS: {service_benefits}\n\n"
            f"LAYOUT: Premium promotional graphic. Service name large and prominent. "
            f"Background gradient using {primary_color} and {secondary_color}. "
            f"2-3 benefit points. 'Prenota ora' call to action. "
            f"Center name '{center_name}' as signature."
        ),
        "stagionale": (
            f"GRAPHIC TYPE: Seasonal post\n"
            f"MAIN MESSAGE (Italian): {concept}\n\n"
            f"LAYOUT: Seasonal mood graphic. Abstract seasonal botanical or geometric elements "
            f"(flowers, leaves, warm/cool tones depending on season). "
            f"Brand colors blended with season palette. "
            f"Main text centered. Center name '{center_name}' as signature."
        ),
        "curiosita": (
            f"GRAPHIC TYPE: 'Did you know?' beauty curiosity post\n"
            f"CURIOSITY (Italian): {concept}\n\n"
            f"LAYOUT: Engaging informational graphic. Large question or fact text as the hero. "
            f"Use a playful but elegant visual element (lightbulb icon, sparkle, or abstract shape). "
            f"Background: soft {bg_color} or light gradient of {primary_color}. "
            f"Center name '{center_name}' as signature at the bottom."
        ),
    }

    prompt = (
        f"Create a professional, publication-ready 1:1 Instagram graphic "
        f"for Italian beauty salon '{center_name}'.\n\n"
        f"{category_design.get(category, category_design['tip_beauty'])}\n\n"
        f"BRAND IDENTITY:\n"
        f"- Visual mood: {visual_style}\n"
        f"- Typography: {typo_style}\n"
        f"- Primary color: {primary_color}\n"
        f"- Secondary color: {secondary_color}\n"
        f"- Accent: {accent_color}\n"
        f"- Background: {bg_color}\n\n"
        f"ABSOLUTE RULES:\n"
        f"- This is a GRAPHIC DESIGN post, NOT a photo\n"
        f"- Do NOT include realistic photos, faces, hands, or skin\n"
        f"- Use abstract shapes, flat illustrations, or typography as the main visual element\n"
        f"- All text visible in the graphic must be in Italian\n"
        f"- Professional beauty brand aesthetic — Canva premium template quality\n"
        f"- Output: 1:1 square, publication-ready"
    )

    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=MODEL_IMAGE,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            ),
        )
        feed_bytes = _extract_image_from_response(response)
        if not feed_bytes:
            logger.error(f"Gemini non ha restituito immagine per ai_graphic/{category}")
            return None, None

        feed_pil = _image_to_pil(feed_bytes)
        story_bytes = _create_story_version(feed_pil)
        logger.info(f"Immagine ai_graphic/{category} generata")
        return feed_bytes, story_bytes

    except Exception as e:
        logger.error(f"Errore generazione immagine ai_graphic ({category}): {e}")
        return None, None


async def generate_ai_graphic_post(tenant: dict) -> dict:
    """
    Genera un post settimanale completamente AI (nessuna foto richiesta).

    Ritorna un dict con tutti i campi per il record social_content,
    più 'feed_bytes' e 'story_bytes' (bytes da caricare su Storage)
    e 'ai_graphic_category' (categoria usata, per aggiornare la rotazione).
    """
    social_profile = tenant.get("social_profile") or {}
    tenant_id = tenant.get("id", "")

    # 1. Categoria corrente dalla rotazione
    category = _get_next_ai_graphic_category(social_profile)
    logger.info(f"[ai_graphic] tenant={tenant_id} categoria={category}")

    # 2. Se spotlight, scegli servizio da mettere in vetrina
    spotlight_service = None
    if category == "spotlight":
        spotlight_service = await _pick_spotlight_service(tenant_id)
        svc_name = spotlight_service.get("name") if spotlight_service else "nessuno"
        logger.info(f"[ai_graphic] spotlight → servizio: {svc_name}")

    # 3. Genera testo
    brand_system_prompt = _get_brand_system_prompt(tenant)
    concept, caption, hashtags = await _generate_ai_graphic_text(
        category=category,
        tenant=tenant,
        spotlight_service=spotlight_service,
        system_prompt=brand_system_prompt,
    )

    # 4. Genera immagine
    feed_bytes, story_bytes = await _generate_ai_graphic_image(
        category=category,
        concept=concept,
        tenant=tenant,
        spotlight_service=spotlight_service,
    )

    return {
        "archetype": "ai_graphic",
        "content_type": "post",
        "material_checklist": [],
        "photos_input": [],
        "caption_text": caption,
        "hashtags": hashtags,
        "selection_rationale": f"Post AI settimanale — categoria: {category}",
        "ai_graphic_category": category,
        "feed_bytes": feed_bytes,
        "story_bytes": story_bytes,
        "service_id": spotlight_service.get("id") if spotlight_service else None,
    }


def _pick_archetype_by_rotation(
    weights: dict[str, int],
    recent_archetypes: list[str],
) -> str:
    """
    Sceglie l'archetype più "in debito" rispetto ai pesi target.

    - weights:           {archetype: target_percent}, somma = 100
    - recent_archetypes: lista degli ultimi N archetypes già usati per questo servizio

    Algoritmo: debt = target% - actual%. Vince il più in debito.
    Se nessuno storico disponibile, vince il peso maggiore.
    """
    if not recent_archetypes:
        return max(weights, key=lambda k: weights[k])

    total = len(recent_archetypes)
    actual_count: dict[str, int] = {a: 0 for a in weights}
    for a in recent_archetypes:
        if a in actual_count:
            actual_count[a] += 1

    debt = {
        a: (weights[a] / 100) - (actual_count[a] / total)
        for a in weights
    }
    return max(debt, key=lambda k: debt[k])


async def _get_recent_archetypes_for_service(
    tenant_id: str,
    service_id: Optional[str],
    limit: int = 8,
) -> list[str]:
    """Interroga social_content per gli ultimi archetype usati per questo servizio."""
    if not service_id:
        return []
    try:
        from src.supabase_client import get_supabase
        sb = get_supabase()
        res = (
            sb.table("social_content")
            .select("archetype")
            .eq("tenant_id", tenant_id)
            .eq("service_id", service_id)
            .not_.is_("archetype", "null")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return [row["archetype"] for row in (res.data or []) if row.get("archetype")]
    except Exception as e:
        logger.warning(f"Storico archetype non disponibile per service {service_id}: {e}")
        return []


async def _get_rules_with_rotation(
    service_name: str,
    service_id: Optional[str],
    tenant_id: str,
) -> dict:
    """
    Determina archetype e checklist per un servizio usando la rotazione.
    Legge lo storico DB → sceglie l'archetype più in debito → ritorna
    {archetype, checklist} compatibile con il resto della pipeline.
    """
    name_lower = service_name.lower()
    cat: Optional[dict] = None
    for cat_data in _SERVICE_RULES_V2.values():
        if any(kw in name_lower for kw in cat_data["keywords"]):
            cat = cat_data
            break
    if cat is None:
        cat = _DEFAULT_RULES_V2

    weights = cat["rotation_weights"]
    archetypes_data = cat["archetypes"]

    recent = await _get_recent_archetypes_for_service(tenant_id, service_id, limit=8)
    chosen = _pick_archetype_by_rotation(weights, recent)

    # Fallback se l'archetype scelto non ha checklist (non dovrebbe mai succedere)
    if chosen not in archetypes_data:
        chosen = next(iter(archetypes_data))

    logger.debug(
        f"Rotazione archetype → servizio='{service_name}' recenti={recent} scelto={chosen}"
    )
    return {"archetype": chosen, "checklist": archetypes_data[chosen]}


# ── Brand system prompt ────────────────────────────────────────────

def _build_brand_system_prompt(tenant: dict) -> str:
    """
    Assembla il system instruction completo del brand da passare a TUTTE
    le chiamate Gemini. Copre identità, persona, voce, contenuto e visual.
    """
    sp = tenant.get("social_profile") or {}
    name = tenant.get("display_name") or sp.get("center_name") or tenant.get("name", "Centro Estetico")
    bio = (tenant.get("bio") or "").strip()

    # ── Identità ──
    tagline = sp.get("tagline") or ""
    city = sp.get("city") or ""
    founded = sp.get("founded_year")
    positioning = sp.get("price_positioning") or "mid-range"
    usp = sp.get("unique_selling_point") or ""
    mission = sp.get("mission") or ""

    # ── Persona ──
    persona_name = sp.get("persona_name") or ""
    persona_age = sp.get("persona_age_range") or "25-45"
    persona_desc = sp.get("persona_description") or "Donne che si prendono cura di sé"
    pain_points = sp.get("persona_pain_points") or []
    desires = sp.get("persona_desires") or []

    # ── Voce ──
    tone = sp.get("tone_of_voice") or "caldo e professionale"
    if isinstance(tone, list):
        tone = ", ".join(tone)
    traits = sp.get("personality_traits") or []
    comm_style = sp.get("communication_style") or "informale, dai del tu"
    emoji_usage = sp.get("emoji_usage") or "moderato (2-3 per post)"
    caption_length = sp.get("caption_length") or "medium"
    cta_style = {
        "link_in_bio": "Prenota ora → link in bio",
        "phone": "Chiamaci per prenotare",
        "whatsapp": "Scrivici su WhatsApp",
        "dm": "Scrivici in DM",
    }.get(sp.get("cta_style") or "link_in_bio", "Prenota ora → link in bio")
    avoid = sp.get("avoid_words") or ["ogni trattamento", "la costanza premia", "risultati visibili"]
    avoid_str = ", ".join(f'"{w}"' for w in avoid)
    signatures = sp.get("signature_phrases") or []
    sig_block = "\n".join(f"  • {p}" for p in signatures) if signatures else "  (nessuna)"
    voice_memo = sp.get("brand_voice_memo") or ""

    # ── Contenuto ──
    pillars = sp.get("content_pillars") or [
        "Risultati dei trattamenti", "Educazione beauty", "Behind the scenes", "Promozioni"
    ]
    content_mix = sp.get("content_mix") or {}
    brand_hashtags = sp.get("brand_hashtags") or []
    niche_hashtags = sp.get("niche_hashtags") or []
    hashtags_str = " ".join(brand_hashtags) if brand_hashtags else "(nessuno)"
    niche_str = " ".join(niche_hashtags) if niche_hashtags else ""
    hashtags_per_post = sp.get("hashtags_per_post") or 10

    # ── Visivo ──
    visual_style = sp.get("visual_style") or sp.get("style") or "minimal"
    photo_style = {
        "bright_natural": "luminose e naturali, luce dalla finestra, colori vivi",
        "warm_moody": "calde e atmosferiche, toni ambrati, ombre morbide",
        "clean_white": "pulite su sfondo bianco, lighting flat, look clinico-professionale",
        "dark_luxury": "scure e lussuose, contrasti forti, atmosfera premium",
    }.get(sp.get("photo_style") or "bright_natural", "luminose e naturali")
    typo_style = {
        "serif_elegant": "serif elegante (es. Playfair Display, Georgia)",
        "sans_modern": "sans-serif moderno (es. Inter, Helvetica)",
        "mixed": "misto: serif per titoli, sans per body",
    }.get(sp.get("typography_style") or "serif_elegant", "serif elegante")
    primary_color = tenant.get("theme_primary_color") or "#6b2d4e"
    secondary_color = tenant.get("theme_secondary_color") or "#c9a0b4"
    accent_color = sp.get("accent_color") or ""
    bg_color = sp.get("background_color") or "#fdf5f0"

    # ── Assembla ──
    lines = [
        f'Sei il social media manager esclusivo di "{name}".',
        "La tua missione: creare contenuti che sembrino scritti da una persona vera che conosce il centro — mai da un robot.",
        "",
        "━━━ IDENTITÀ ━━━",
        f"Centro: {name}",
    ]
    if city:
        identity_line = f"Città: {city}"
        if founded:
            identity_line += f" — aperto dal {founded}"
        lines.append(identity_line)
    if tagline:
        lines.append(f'Tagline: "{tagline}"')
    if bio:
        lines.append(f"Descrizione: {bio}")
    if mission:
        lines.append(f"Mission: {mission}")
    lines.append(f"Posizionamento: {positioning}")
    if usp:
        lines.append(f"Unicità: {usp}")

    lines += ["", "━━━ CLIENTE IDEALE ━━━"]
    persona_header = f"{persona_name} — " if persona_name else ""
    lines.append(f"{persona_header}{persona_age} anni — {persona_desc}")
    if pain_points:
        lines.append("Frustrata da: " + ", ".join(pain_points))
    if desires:
        lines.append("Vuole: " + ", ".join(desires))

    lines += [
        "",
        "━━━ VOCE DEL BRAND ━━━",
        f"Personalità: {', '.join(traits) if traits else tone}",
        f"Tono: {tone}",
        f"Stile: {comm_style}",
        f"Emoji: {emoji_usage}",
        f"Lunghezza caption: {caption_length} (~{'50' if caption_length=='short' else '100' if caption_length=='medium' else '150'} parole)",
        f"CTA standard: \"{cta_style}\"",
        "",
        "Frasi tipiche del brand (ispiratene, non copiare letteralmente):",
        sig_block,
        "",
        f"NON usare MAI: {avoid_str}",
        'NON usare mai: "ogni trattamento è unico", "la cura parte da te", "risultati che parlano da soli", "benessere a 360°"',
    ]

    if voice_memo:
        lines += ["", "Nota dalla titolare:", f'  "{voice_memo}"']

    lines += [
        "",
        "━━━ PILASTRI EDITORIALI ━━━",
    ]
    for p in pillars:
        pct = ""
        key_map = {"risultati": "results", "educaz": "education", "dietro": "behind_scenes", "promo": "promo"}
        for k, v in key_map.items():
            if k in p.lower() and v in content_mix:
                pct = f" ({content_mix[v]}%)"
                break
        lines.append(f"  • {p}{pct}")

    lines += [
        "",
        f"━━━ HASHTAG ({hashtags_per_post} per post) ━━━",
        f"Fissi (sempre): {hashtags_str}",
    ]
    if niche_str:
        lines.append(f"Di nicchia (scegli pertinenti): {niche_str}")

    lines += [
        "",
        "━━━ STILE VISIVO ━━━",
        f"Grafico: {visual_style}",
        f"Foto: {photo_style}",
        f"Tipografia: {typo_style}",
        f"Colori: {primary_color} (primario) · {secondary_color} (secondario)" +
        (f" · {accent_color} (accento)" if accent_color else "") +
        f" · {bg_color} (sfondo)",
    ]

    return "\n".join(lines)


# ── Brand onboarding conversazionale ──────────────────────────────

async def run_brand_chat_turn(
    messages: list[dict],
    existing_profile: dict,
    tenant_name: str,
) -> tuple[str, bool]:
    """
    Esegue un turno dell'intervista brand.
    Returns (reply_pulito, is_complete).
    """
    profile_context = ""
    parts = []
    if existing_profile.get("city"):
        parts.append(f"Città: {existing_profile['city']}")
    if existing_profile.get("tagline"):
        parts.append(f"Tagline: {existing_profile['tagline']}")
    tone = existing_profile.get("tone_of_voice")
    if tone:
        parts.append(f"Tono attuale: {', '.join(tone) if isinstance(tone, list) else tone}")
    if existing_profile.get("target_description"):
        parts.append(f"Target: {existing_profile['target_description']}")
    if parts:
        profile_context = (
            "Dati già disponibili (usali come contesto, non chiedere di nuovo):\n"
            + "\n".join(f"- {p}" for p in parts)
            + "\n\n"
        )

    system = (
        f'Sei un esperto di brand identity per centri estetici italiani.\n'
        f'Conduci un\'intervista in italiano per capire il brand di "{tenant_name}".\n\n'
        f'{profile_context}'
        f'Obiettivo: raccogliere informazioni autentiche per creare un brand prompt efficace.\n\n'
        f'Schema (max 5-6 domande totali, UNA alla volta):\n'
        f'1. Prima risposta: presentati in 2 righe, poi chiedi subito cosa rende unico il centro — non tecnicamente, ma come "anima".\n'
        f'2. Chi è la cliente tipica? Cosa la fa tornare?\n'
        f'3. Come parlate alle clienti? Dammi un esempio di frase che usereste su Instagram.\n'
        f'4. Cosa NON volete che si dica mai del vostro centro?\n'
        f'5. C\'è un servizio o momento speciale che volete valorizzare?\n'
        f'6. (Solo se mancano info cruciali) Una domanda libera.\n\n'
        f'Dopo la 5a-6a risposta: di\' "Perfetto, ho tutto quello che mi serve."\n'
        f'Poi aggiungi ESATTAMENTE il tag (incluse parentesi quadre): [ONBOARDING_COMPLETE]\n\n'
        f'Tono: professionale ma caldo. Una domanda alla volta. Niente elenchi puntati nelle domande.\n'
        f'Rispondi SOLO con il messaggio per il titolare.'
    )

    contents = []
    # Se no messaggi → messaggio fittizio per far partire l'agente
    if not messages:
        contents = [types.Content(role="user", parts=[types.Part(text="Inizia l'intervista.")])]
    else:
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    response = await asyncio.to_thread(
        _get_client().models.generate_content,
        model=MODEL_TEXT,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    reply = (response.text or "").strip()
    is_complete = "[ONBOARDING_COMPLETE]" in reply
    clean_reply = reply.replace("[ONBOARDING_COMPLETE]", "").strip()
    return clean_reply, is_complete


async def generate_prompt_from_conversation(
    messages: list[dict],
    existing_profile: dict,
    tenant: dict,
) -> str:
    """Genera brand_system_prompt completo dalla conversazione di onboarding."""
    tenant_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    primary_color = tenant.get("theme_primary_color") or "#6b2d4e"
    secondary_color = tenant.get("theme_secondary_color") or "#c9a0b4"
    hashtags = existing_profile.get("brand_hashtags") or []
    hashtags_str = " ".join(hashtags) if hashtags else "(da definire)"

    conversation = "\n".join(
        f"{'Agente' if m['role'] == 'assistant' else 'Titolare'}: {m['content']}"
        for m in messages
        if m["role"] != "system"
    )

    system = (
        f'Sei un esperto copywriter di brand identity per centri estetici italiani.\n'
        f'Crea un brand_system_prompt per "{tenant_name}" basandoti sull\'intervista.\n\n'
        f'Il prompt sarà usato come system instruction da un AI per generare contenuti social.\n'
        f'Deve essere in italiano, concreto, catturare l\'autenticità del brand.\n\n'
        f'Struttura con separatori ━━━:\n'
        f'- IDENTITÀ\n- CLIENTE IDEALE\n- VOCE DEL BRAND\n- COSA EVITARE\n- PILASTRI EDITORIALI\n'
        f'- HASHTAG (includi: {hashtags_str})\n'
        f'- STILE VISIVO (colori: {primary_color} primario, {secondary_color} secondario)\n\n'
        f'Prima riga obbligatoria: \'Sei il social media manager esclusivo di "{tenant_name}".\'\n'
        f'Scrivi come se istruissi un collega. Diretto, specifico, autentico.\n'
        f'Rispondi SOLO con il prompt, nessun commento aggiuntivo.'
    )

    response = await asyncio.to_thread(
        _get_client().models.generate_content,
        model=MODEL_TEXT,
        contents=f"INTERVISTA:\n{conversation}\n\nCrea il brand prompt completo.",
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return (response.text or "").strip()


def _get_brand_system_prompt(tenant: dict) -> str:
    """Usa brand_system_prompt personalizzato se esiste, altrimenti build da template."""
    sp = tenant.get("social_profile") or {}
    custom = (sp.get("brand_system_prompt") or "").strip()
    return custom if custom else _build_brand_system_prompt(tenant)


async def suggest_prompt_modification(
    current_prompt: str,
    instruction: str,
    tenant_name: str,
) -> str:
    """Chiama Gemini per modificare brand_system_prompt seguendo l'istruzione."""
    system = (
        f'Sei un esperto di brand identity per centri estetici.\n'
        f'Compito: modificare il system prompt del brand "{tenant_name}" '
        f'seguendo l\'istruzione ricevuta.\n\n'
        f'Regole:\n'
        f'- Mantieni struttura e sezioni del prompt originale\n'
        f'- Applica SOLO la modifica richiesta, non stravolgere il resto\n'
        f'- Il prompt risultante deve essere completo e autonomo\n'
        f'- Rispondi SOLO con il prompt modificato, nessun commento aggiuntivo'
    )
    user_msg = (
        f"PROMPT ATTUALE:\n{current_prompt}\n\n"
        f"ISTRUZIONE MODIFICA:\n{instruction}\n\n"
        f"Restituisci il prompt modificato completo."
    )
    response = await asyncio.to_thread(
        _get_client().models.generate_content,
        model=MODEL_TEXT,
        contents=user_msg,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return (response.text or "").strip()


# ── 1. Selezione e pianificazione settimanale ──────────────────────

async def select_and_plan_week(
    appointments: list[dict],
    tenant: dict,
    week_start_str: str,
    week_end_str: str,
    existing_appointment_ids: list[str],
) -> list[dict]:
    """
    Seleziona gli appuntamenti e crea il piano editoriale.

    - Python decide archetype e schema foto (deterministico, affidabile)
    - Gemini personalizza le istruzioni foto per il servizio specifico (dinamico)
    - Gemini genera caption e hashtag (creativo)
    """
    social_profile = tenant.get("social_profile") or {}
    content_frequency = social_profile.get("content_frequency", 3)
    tone_of_voice = social_profile.get("tone_of_voice", "caldo e professionale")
    if isinstance(tone_of_voice, list):
        tone_of_voice = ", ".join(tone_of_voice)
    brand_keywords = social_profile.get("brand_keywords", [])
    center_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")

    # Servizi da escludere completamente (zone intime, non fotografabili)
    ESCLUDI = ["epilazione inguine", "epilazione bikini", "ceretta inguine", "ceretta bikini"]

    # Filtra candidati
    candidati = []
    for a in appointments:
        appt_id = a.get("id")
        if appt_id in existing_appointment_ids:
            continue
        service = a.get("service") or {}
        service_name = service.get("name", "")
        if any(k.lower() in service_name.lower() for k in ESCLUDI):
            continue
        candidati.append({
            "id": appt_id,
            "giorno": a.get("start_at", "")[:10],
            "servizio": service_name,
            "service_id": service.get("id"),  # per la rotazione archetype
        })

    if not candidati:
        logger.info("Nessun appuntamento candidato per questa settimana")
        return []

    # Distribuisce i candidati su giorni diversi prima di limitare al content_frequency.
    # Senza questo, con molti appuntamenti tutti il lunedì il [:content_frequency]
    # prenderebbe solo lunedì anche se ci sono appuntamenti in altri giorni.
    from collections import defaultdict as _dd
    _per_giorno: dict[str, list] = _dd(list)
    for c in candidati:
        _per_giorno[c["giorno"]].append(c)
    _distribuiti: list = []
    _giorni = list(_per_giorno.keys())  # già ordinati perché appointments è ASC
    _idx = 0
    while len(_distribuiti) < content_frequency:
        _added = False
        for g in _giorni:
            if len(_distribuiti) >= content_frequency:
                break
            if _idx < len(_per_giorno[g]):
                _distribuiti.append(_per_giorno[g][_idx])
                _added = True
        if not _added:
            break
        _idx += 1
    candidati = _distribuiti

    # Assembla il system prompt del brand una volta sola
    brand_system_prompt = _get_brand_system_prompt(tenant)

    # Per ogni candidato: rotazione archetype, Gemini personalizza istruzioni + caption
    results = []
    tenant_id = tenant.get("id", "")
    for c in candidati:
        service_name = c["servizio"]
        rules = await _get_rules_with_rotation(
            service_name=service_name,
            service_id=c.get("service_id"),
            tenant_id=tenant_id,
        )

        # Gemini personalizza le istruzioni foto (con fallback alle istruzioni base)
        checklist = await _personalize_checklist_instructions(
            service_name=service_name,
            archetype=rules["archetype"],
            base_checklist=rules["checklist"],
            system_prompt=brand_system_prompt,
        )

        # Gemini genera caption e hashtag
        caption, hashtags = await _generate_caption_and_hashtags(
            service_name=service_name,
            archetype=rules["archetype"],
            system_prompt=brand_system_prompt,
        )

        results.append({
            "appointment_id": c["id"],
            "service_name": service_name,
            "archetype": rules["archetype"],
            "content_type": "post",
            "rationale": f"Servizio {service_name} del {c['giorno']}",
            "material_checklist": checklist,
            "caption_text": caption,
            "hashtags": hashtags,
        })

    logger.info(f"Piano creato per {len(results)} contenuti")
    return results


async def _personalize_checklist_instructions(
    service_name: str,
    archetype: str,
    base_checklist: list[dict],
    system_prompt: str = "",
) -> list[dict]:
    """
    Gemini personalizza le istruzioni foto per il servizio specifico.
    Python ha già deciso quante foto e di che tipo — Gemini scrive solo
    le istruzioni concrete adatte a quel trattamento esatto.
    Fallback: istruzioni base se Gemini fallisce.
    """
    items_desc = "\n".join([
        f"- Foto {i+1}: id='{item['id']}', label='{item['label']}'"
        for i, item in enumerate(base_checklist)
    ])

    prompt = f"""Sei un fotografo esperto di beauty content per Instagram italiano.

Scrivi istruzioni pratiche per fotografare questo trattamento estetico:

SERVIZIO: {service_name}
TIPO DI POST: {archetype}
FOTO DA SCATTARE:
{items_desc}

Per OGNI foto scrivi 2-3 frasi che spiegano:
- Come posizionare il soggetto (mano, viso, corpo, macchinario...)
- Dove mettere la luce (finestra, soffitto, flash...)
- Che angolazione o distanza usare

Le istruzioni devono essere SPECIFICHE per "{service_name}", pratiche e comprensibili
anche per chi non è fotografo. Niente frasi generiche come "buona luce" o "ambiente curato".

Rispondi SOLO con JSON:
{{
  "instructions": [
    {{"id": "...", "instructions": "..."}},
    ...
  ]
}}"""

    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=MODEL_TEXT,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=0.4,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text
        logger.debug(f"Risposta Gemini checklist per '{service_name}': {raw_text!r}")

        if not raw_text:
            candidates = getattr(response, "candidates", None) or []
            finish_reason = getattr(candidates[0], "finish_reason", "N/A") if candidates else "N/A"
            logger.warning(
                f"Gemini ha restituito risposta vuota/None per checklist '{service_name}'. "
                f"Finish reason: {finish_reason}"
            )
            return base_checklist

        data = _parse_json_response(raw_text)
        items_map = {item["id"]: item["instructions"] for item in data.get("instructions", [])}
        logger.debug(f"items_map per '{service_name}': {items_map}")

        if not items_map:
            logger.warning(f"Gemini non ha generato istruzioni per '{service_name}'. JSON: {data}")
            return base_checklist

        # Merge: struttura base Python + istruzioni personalizzate Gemini
        result = []
        for base_item in base_checklist:
            item_copy = dict(base_item)
            if base_item["id"] in items_map and items_map[base_item["id"]].strip():
                item_copy["instructions"] = items_map[base_item["id"]]
            else:
                logger.warning(
                    f"Istruzione mancante/vuota per item '{base_item['id']}' "
                    f"(servizio: {service_name}). Disponibili: {list(items_map.keys())}"
                )
            result.append(item_copy)
        return result
    except Exception as e:
        logger.exception(f"Errore personalizzazione checklist per '{service_name}': {e}")
        return base_checklist  # fallback istruzioni base


async def _generate_caption_and_hashtags(
    service_name: str,
    archetype: str,
    system_prompt: str = "",
) -> tuple[str, list[str]]:
    """
    Chiama Gemini per caption e hashtag.
    Il contesto brand (tono, stile, hashtag fissi, parole vietate) viene
    dal system_prompt assemblato da _build_brand_system_prompt().
    """
    archetype_hint = {
        "before_after": "mostra la trasformazione prima/dopo",
        "editorial": "valorizza il risultato estetico del trattamento",
        "educational": "spiega il trattamento in modo semplice",
        "behind_scenes": "mostra il processo e le mani al lavoro",
        "promo": "invita a prenotare con un'offerta",
    }.get(archetype, "valorizza il trattamento")

    prompt = f"""Scrivi una caption Instagram e hashtag per questo contenuto:

SERVIZIO: {service_name}
TIPO DI POST: {archetype_hint}

REGOLE CAPTION:
- Prima riga: menziona il nome esatto del servizio "{service_name}"
- Descrivi il risultato specifico di QUESTO servizio, mai frasi generiche
- CTA finale coerente col tono del brand
- Rispetta RIGOROSAMENTE le regole su emoji, parole vietate e tono del system instruction
- Max 100 parole

HASHTAG: 8-12 hashtag. Includi SEMPRE gli hashtag fissi del brand (nel system instruction)
più hashtag specifici per "{service_name}".

Rispondi SOLO con JSON valido:
{{
  "caption": "testo caption...",
  "hashtags": ["hashtag1", "hashtag2"]
}}"""

    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=MODEL_TEXT,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=0.7,
                response_mime_type="application/json",
            ),
        )
        data = _parse_json_response(response.text)
        return data.get("caption", ""), data.get("hashtags", [])
    except Exception as e:
        logger.error(f"Errore generazione caption per {service_name}: {e}")
        return f"Scopri il trattamento {service_name}! Prenota ora → link in bio", []


# ── Creative briefs per archetype ─────────────────────────────────
#
# Invece di template fissi da seguire, Gemini riceve:
#   1. Un OBIETTIVO creativo per l'archetype (cosa deve comunicare)
#   2. ISPIRAZIONE con possibili approcci (non istruzioni rigide)
#   3. VINCOLI di brand (colori, font, privacy) — questi sì rigidi
#
# Gemini ha libertà creativa su layout, composizione, elementi grafici,
# posizione testi. I post usciranno visivamente diversi tra loro.



def _get_treatment_area(service_name: str) -> str:
    """Restituisce la descrizione della zona protetta in base al servizio."""
    s = (service_name or "").lower()
    if any(k in s for k in ["ciglia", "lash", "lamina", "extension"]):
        return "eye area, lashes, eyelids, and brows"
    elif any(k in s for k in ["sopracciglia", "brow", "henna"]):
        return "brow area and eye zone"
    elif any(k in s for k in ["unghie", "nail", "smalto", "manicure", "pedicure", "gel"]):
        return "nails, fingertips, and hands"
    elif any(k in s for k in ["viso", "face", "pulizia", "peeling", "dermapen", "filler"]):
        return "skin texture, face detail, and pore visibility"
    elif any(k in s for k in ["laser", "epilazione", "diodo"]):
        return "treated skin area and surrounding skin"
    elif any(k in s for k in ["massaggio", "corpo", "body", "rimodellante"]):
        return "body contour and skin detail"
    else:
        return "the main treatment result area"


# ── 2. Generazione brief visivo ────────────────────────────────────

async def generate_visual_brief(
    content_record: dict,
    tenant: dict,
) -> str:
    """
    Analizza le foto caricate e descrive IN ITALIANO SEMPLICE
    cosa creerà — mostrato all'estetista PRIMA di generare l'immagine.
    """
    photos = content_record.get("photos_input") or []
    archetype = content_record.get("archetype", "editorial")
    # service_name viene dal join service:services(name) — NON dal campo diretto
    service_data = content_record.get("service") or {}
    service_name = service_data.get("name") or content_record.get("service_name") or ""
    service_desc = service_data.get("descrizione_breve") or ""
    service_benefits = service_data.get("benefici") or ""
    notes = content_record.get("estetista_notes") or ""
    consent = content_record.get("client_consent", "details_only")

    brand_system_prompt = _get_brand_system_prompt(tenant)
    social_profile = tenant.get("social_profile") or {}
    center_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    primary_color = tenant.get("theme_primary_color") or "#6b2d4e"
    secondary_color = tenant.get("theme_secondary_color") or "#c9a0b4"

    consent_instruction = {
        "with_face": "Puoi mostrare il viso della cliente.",
        "details_only": "Mostra solo mani, dettagli o zone specifiche, NON il viso.",
        "no_client": "Non mostrare la cliente. Usa solo prodotti e ambiente.",
    }.get(consent, "Mostra solo dettagli, no viso.")

    pil_images = []
    for url in photos[:3]:
        try:
            img = await _download_image(url)
            pil_images.append(img)
        except Exception as e:
            logger.warning(f"Impossibile scaricare foto {url}: {e}")

    sp = social_profile  # alias locale
    accent_color = sp.get("accent_color") or secondary_color
    bg_color = sp.get("background_color") or "#fdf5f0"
    visual_style = sp.get("visual_style") or sp.get("style") or "minimal"
    photo_style = sp.get("photo_style") or "bright_natural"
    typo_style = sp.get("typography_style") or "serif_elegant"

    photo_style_desc = {
        "bright_natural": "luminosa e naturale, luce calda dalla finestra, colori vivaci",
        "warm_moody":     "calda e atmosferica, toni ambrati, ombre morbide",
        "clean_white":    "pulita e professionale, sfondo bianco, luce piatta",
        "dark_luxury":    "scura e lussuosa, contrasti forti, atmosfera premium",
    }.get(photo_style, "luminosa e naturale")

    typo_desc = {
        "serif_elegant": "font serif elegante (stile Playfair Display)",
        "sans_modern":   "font sans-serif moderno e pulito",
        "mixed":         "titolo in serif, testo corpo in sans-serif",
    }.get(typo_style, "font serif elegante")

    # Obiettivo creativo dell'archetype (in italiano per il brief all'estetista)
    archetype_goal_it = {
        "before_after":  "Mostrare la trasformazione prima/dopo in modo che l'impatto si veda subito",
        "editorial":     "Creare un'immagine aspirazionale e curata, stile editoriale beauty",
        "educational":   "Spiegare il trattamento in modo chiaro e invitante — si legge in 3 secondi",
        "behind_scenes": "Catturare un momento autentico del lavoro in cabina, minimal e genuino",
        "promo":         "Creare un post d'impatto che invoglia subito a prenotare",
    }.get(archetype, "Creare un post bello e coerente col brand")

    service_context = ""
    if service_desc:
        service_context += f"Descrizione servizio: {service_desc}\n"
    if service_benefits:
        service_context += f"Benefici: {service_benefits}\n"

    prompt = f"""Stai per creare un post per: {center_name}
Servizio: {service_name}
{service_context}Tipo di post: {archetype}
{f"Note dell'estetista: {notes}" if notes else ""}
Regola privacy: {consent_instruction}

PALETTE BRAND (descrivi i colori con parole, non codici):
- Primario: {primary_color}
- Secondario: {secondary_color}
- Accento: {accent_color}
- Sfondo: {bg_color}

STILE FOTO: {photo_style_desc}
TIPOGRAFIA: {typo_desc}

OBIETTIVO: {archetype_goal_it}

Guarda le foto e descrivi IN ITALIANO SEMPLICE il post che creerai.
Hai libertà creativa completa su layout, composizione ed elementi grafici.

📐 IDEA
(1-2 frasi: che composizione hai scelto e perché funziona per queste foto)

🎨 STILE
(Colori in parole normali. Atmosfera. Elementi grafici decorativi se presenti)

✍️ TESTI IN GRAFICA
(SOLO i testi che appariranno visivamente sulla grafica: nome servizio, nome centro, eventuali label come PRIMA/DOPO o titoli. NON scrivere la caption del post — quella è già gestita separatamente)

✨ EFFETTO
(Che sensazione trasmette? Perché ferma chi scorre il feed?)

Regole:
- Italiano semplice, zero termini tecnici
- I colori in parole normali (es. "viola profondo", "crema caldo", "rosa antico")
- Max 130 parole
- SOLO le foto fornite — non inventare"""

    try:
        client = _get_client()
        contents: list = [prompt] + pil_images

        response = await client.aio.models.generate_content(
            model=MODEL_TEXT,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=brand_system_prompt,
                temperature=0.7,
            ),
        )
        brief = response.text.strip()
        logger.info(f"Brief visivo generato per content {content_record.get('id')}")
        return brief
    except Exception as e:
        logger.error(f"Errore generazione brief: {e}")
        return "Non è stato possibile generare il brief visivo. Riprova."


# ── 3. Generazione immagine ────────────────────────────────────────

# Direzioni creative per le 3 varianti
async def generate_image_variants(
    content_record: dict,
    tenant: dict,
) -> list[dict]:
    """
    Genera 3 varianti in parallelo con lo stesso prompt — il modello produce
    output diversi naturalmente per la sua natura stocastica.
    """
    tasks = [generate_image(content_record, tenant) for _ in range(3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    variants = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Variante {i+1} fallita: {result}")
            continue
        if not result or result == (None, None):
            continue
        feed_bytes, story_bytes = result
        if feed_bytes:
            variants.append({
                "index": i,
                "direction": f"variante_{i+1}",
                "feed_bytes": feed_bytes,
                "story_bytes": story_bytes,
            })

    logger.info(f"Generate {len(variants)}/3 varianti per content {content_record.get('id')}")
    return variants


async def generate_image(
    content_record: dict,
    tenant: dict,
) -> tuple[Optional[bytes], Optional[bytes]]:
    """
    Genera la grafica finale usando gemini-3-pro-image-preview.
    Segue esclusivamente il visual_brief approvato dall'utente.
    """
    photos = content_record.get("photos_input") or []
    archetype = content_record.get("archetype", "editorial")
    # service_name e dati servizio dal join service:services(...)
    service_data = content_record.get("service") or {}
    service_name = service_data.get("name") or content_record.get("service_name") or ""
    service_desc = service_data.get("descrizione_breve") or ""
    service_benefits = service_data.get("benefici") or ""
    brief_override = content_record.get("visual_brief_override")
    brief = brief_override or content_record.get("visual_brief") or ""
    logger.info(
        f"Brief usato per generazione immagine (content {content_record.get('id')}): "
        f"{'OVERRIDE' if brief_override else 'originale'} — {brief[:120]!r}..."
    )

    brand_system_prompt = _get_brand_system_prompt(tenant)
    social_profile = tenant.get("social_profile") or {}
    center_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    primary_color = tenant.get("theme_primary_color") or "#6b2d4e"
    secondary_color = tenant.get("theme_secondary_color") or "#c9a0b4"
    consent = content_record.get("client_consent", "details_only")

    consent_instruction = {
        "with_face": "You may show the client's face.",
        "details_only": "Show only hands or specific treated areas. Do NOT show the client's face.",
        "no_client": "Do not show any client. Use only products and environment.",
    }.get(consent, "Show only details, no face.")

    pil_images = []
    for url in photos[:3]:
        try:
            img = await _download_image(url)
            pil_images.append(img)
        except Exception as e:
            logger.warning(f"Impossibile scaricare foto {url}: {e}")

    sp = social_profile  # alias locale
    accent_color = sp.get("accent_color") or secondary_color
    bg_color = sp.get("background_color") or "#fdf5f0"

    service_text = f'"{service_name}"' if service_name else ""
    brief_section = (
        f"Visual brief approved by the user — follow it precisely:\n{brief}\n\n"
        if brief else ""
    )

    try:
        client = _get_client()

        # ── STEP 1: Analisi spaziale (testo, veloce) ─────────────────────
        # Prima di generare, chiediamo al modello di analizzare esattamente
        # dove si trova il soggetto e dove ci sono zone vuote.
        # Questo segue la best practice Google "step-by-step instructions"
        # per task complessi con vincoli di posizionamento.
        zone_analysis = ""
        if pil_images:
            analysis_prompt = (
                "You are an art director analyzing a beauty photo to plan a graphic overlay.\n\n"
                "Think like a graphic designer — not about avoiding areas, but about what works visually.\n\n"
                "Provide a concise analysis in three parts:\n\n"
                "1. COMPOSITION: What is the main subject, where is it, and what is the visual weight distribution?\n\n"
                "2. GRAPHIC OPPORTUNITIES: Which areas of the photo could host text or graphic elements "
                "in a way that looks intentional and elegant? Consider: areas with uniform color or texture "
                "where text would be legible, areas where a graphic element would balance the composition, "
                "and even areas ON the subject if the contrast and space allow it without disturbing the result.\n\n"
                "3. DESIGNER RECOMMENDATION: Where would a professional graphic designer place the text "
                "and brand elements to create the most impactful, cohesive result? "
                "Be specific and justify the choice in visual/compositional terms.\n\n"
                "Be concise and direct."
            )
            try:
                analysis_response = await client.aio.models.generate_content(
                    model=MODEL_TEXT,
                    contents=[analysis_prompt] + pil_images,
                    config=types.GenerateContentConfig(temperature=0.1),
                )
                zone_analysis = analysis_response.text.strip()
                logger.debug(f"Analisi zone per content {content_record.get('id')}: {zone_analysis[:150]}...")
            except Exception as e:
                logger.warning(f"Analisi zone fallita, procedo senza: {e}")

        # ── STEP 2: Generazione immagine con contesto spaziale esplicito ──
        placement_section = (
            f"ART DIRECTION ANALYSIS — use this to inform your design decisions:\n"
            f"{zone_analysis}\n\n"
            if zone_analysis else
            "Study the photo composition carefully before designing.\n\n"
        )

        prompt = (
            f"Create a publication-ready 1:1 Instagram post for an Italian beauty center.\n\n"

            f"{placement_section}"
            f"Brand colors: primary {primary_color} · secondary {secondary_color} · "
            f"accent {accent_color} · background {bg_color}\n\n"

            + brief_section +

            f"The visual brief above is the ONLY source of truth for what TEXT to show in the graphic. "
            f"Do not add any text, brand names, service names, or labels beyond what the brief explicitly specifies.\n\n"

            f"Think and design like a professional graphic designer:\n"
            f"You have full creative freedom on composition, typography, graphic elements, and style. "
            f"Place text and graphics wherever they create the strongest, most cohesive visual result — "
            f"including on or near the subject if it works compositionally. "
            f"What matters is that the design feels intentional, elegant, and on-brand.\n\n"

            f"One absolute rule: the photo fills the entire canvas edge-to-edge — "
            f"never shrink, frame, or letterbox it.\n\n"

            f"The provided photo is the hero of the composition — do NOT alter, retouch, "
            f"regenerate, or modify the main subject in any way (faces, hands, nails, lashes, "
            f"skin, treated areas). You may work creatively on background areas, add text, "
            f"graphic elements, color overlays, or subtle texture — but the subject must remain "
            f"exactly as in the original photo.\n\n"

            f"Output: 1:1 square, publication-ready."
        )

        contents: list = [prompt] + pil_images

        response = await client.aio.models.generate_content(
            model=MODEL_IMAGE,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="1:1",
                ),
            ),
        )

        feed_bytes = _extract_image_from_response(response)
        if not feed_bytes:
            logger.error("Gemini non ha restituito immagine")
            return None, None

        feed_pil = _image_to_pil(feed_bytes)
        story_bytes = _create_story_version(feed_pil)

        logger.info(f"Immagine generata per content {content_record.get('id')}")
        return feed_bytes, story_bytes

    except Exception as e:
        logger.error(f"Errore generazione immagine: {e}")
        return None, None
