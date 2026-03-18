"""
Logica AI del modulo social — tutte le chiamate Gemini.

Tre funzioni principali:
1. select_and_plan_week()   → sceglie appuntamenti; Python decide archetype+checklist,
                              Gemini fa solo caption+hashtag
2. generate_visual_brief()  → analizza foto e descrive cosa creerà (PRIMA di generare)
3. generate_image()         → genera la grafica finale con gemini-3-pro-image-preview
"""

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


async def _download_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")


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


# ── Knowledge base: Python decide archetype e checklist ────────────

# Lista keyword → regole (archetype, checklist items)
# Ogni keyword viene cercata nel nome del servizio (case-insensitive).
# L'ordine conta: la prima regola che fa match vince.
_SERVICE_RULES = [
    # ── Ciglia ──
    {
        "keywords": ["laminazione ciglia", "extension ciglia", "lifting ciglia", "ciglia"],
        "archetype": "before_after",
        "checklist": [
            {
                "id": "before",
                "label": "Foto ciglia PRIMA",
                "instructions": (
                    "Fotografa l'occhio aperto senza mascara. "
                    "Tieni il telefono vicino al viso con buona luce dalla finestra. "
                    "Stesso angolo che userai per la foto DOPO."
                ),
                "required": True,
            },
            {
                "id": "after",
                "label": "Foto ciglia DOPO",
                "instructions": (
                    "Stesso angolo della foto PRIMA, subito dopo il trattamento. "
                    "Occhi aperti, luce dalla finestra."
                ),
                "required": True,
            },
        ],
    },
    # ── Sopracciglia ──
    {
        "keywords": ["microblading", "nanoblading", "laminazione sopracciglia",
                     "tinta sopracciglia", "sopracciglia"],
        "archetype": "before_after",
        "checklist": [
            {
                "id": "before",
                "label": "Foto sopracciglia PRIMA",
                "instructions": (
                    "Fotografa le sopracciglia da davanti con il viso struccato. "
                    "Tieni il telefono all'altezza degli occhi, buona luce dalla finestra."
                ),
                "required": True,
            },
            {
                "id": "after",
                "label": "Foto sopracciglia DOPO",
                "instructions": (
                    "Stesso angolo della foto PRIMA, subito dopo il trattamento. "
                    "Luce dalla finestra, stessa espressione."
                ),
                "required": True,
            },
        ],
    },
    # ── Semipermanente / Smalto / Manicure / Pedicure / Nail art ──
    {
        "keywords": ["semipermanente", "smalto", "nail art", "ricostruzione unghie",
                     "manicure", "pedicure", "unghie"],
        "archetype": "editorial",
        "checklist": [
            {
                "id": "result",
                "label": "Foto risultato unghie",
                "instructions": (
                    "Appoggia la mano (o il piede) su un asciugamano bianco o sul lettino. "
                    "Tieni il telefono a una spanna dall'unghia e scatta dall'alto. "
                    "Cerca luce dalla finestra, non quella del soffitto."
                ),
                "required": True,
            },
        ],
    },
    # ── Laser / Luce pulsata ──
    {
        "keywords": ["laser", "luce pulsata", "epilazione laser"],
        "archetype": "educational",
        "checklist": [
            {
                "id": "treatment",
                "label": "Foto durante il trattamento",
                "instructions": (
                    "Fotografa il macchinario mentre viene usato sulla pelle "
                    "(zona gambe o ascelle, non zone intime). "
                    "Tieni il telefono a circa 30 cm di distanza, luce naturale."
                ),
                "required": True,
            },
        ],
    },
    # ── Pulizia viso / Trattamenti viso ──
    {
        "keywords": ["pulizia viso", "idratazione viso", "trattamento viso",
                     "peeling", "viso"],
        "archetype": "before_after",
        "checklist": [
            {
                "id": "before",
                "label": "Foto viso PRIMA",
                "instructions": (
                    "Foto del viso in luce naturale, senza filtri o trucco. "
                    "Tieni il telefono all'altezza del viso, sguardo neutro."
                ),
                "required": True,
            },
            {
                "id": "after",
                "label": "Foto viso DOPO",
                "instructions": (
                    "Stesso angolo della foto PRIMA, subito dopo il trattamento. "
                    "Luce dalla finestra, viso rilassato."
                ),
                "required": True,
            },
        ],
    },
    # ── Massaggio / Corpo ──
    {
        "keywords": ["massaggio", "drenante", "pressoterapia", "cavitazione",
                     "radiofrequenza", "mesoterapia"],
        "archetype": "behind_scenes",
        "checklist": [
            {
                "id": "treatment",
                "label": "Foto durante il trattamento",
                "instructions": (
                    "Fotografa le mani dell'operatrice al lavoro, "
                    "o il macchinario sulla zona trattata. "
                    "Cerca luce calda e ambiente ordinato."
                ),
                "required": True,
            },
        ],
    },
]

# Fallback per servizi non riconosciuti
_DEFAULT_RULES = {
    "archetype": "editorial",
    "checklist": [
        {
            "id": "result",
            "label": "Foto del risultato",
            "instructions": (
                "Fotografa il risultato del trattamento in luce naturale. "
                "Tieni il telefono vicino al soggetto e scatta con calma. "
                "Cerca luce dalla finestra."
            ),
            "required": True,
        }
    ],
}


def _get_service_rules(service_name: str) -> dict:
    """Restituisce archetype e checklist in base al nome del servizio."""
    name_lower = service_name.lower()
    for rule in _SERVICE_RULES:
        if any(kw in name_lower for kw in rule["keywords"]):
            return rule
    return _DEFAULT_RULES


# ── 1. Selezione e pianificazione settimanale ──────────────────────

def select_and_plan_week(
    appointments: list[dict],
    tenant: dict,
    week_start_str: str,
    week_end_str: str,
    existing_appointment_ids: list[str],
) -> list[dict]:
    """
    Seleziona gli appuntamenti e crea il piano editoriale.

    - Python decide archetype e checklist (mapping deterministico)
    - Gemini decide solo caption e hashtag (creatività)
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
        })

    if not candidati:
        logger.info("Nessun appuntamento candidato per questa settimana")
        return []

    # Limita al content_frequency
    candidati = candidati[:content_frequency]

    # Per ogni candidato: Python assegna archetype+checklist, Gemini fa caption+hashtag
    results = []
    for c in candidati:
        service_name = c["servizio"]
        rules = _get_service_rules(service_name)

        # Gemini genera solo caption e hashtag
        caption, hashtags = _generate_caption_and_hashtags(
            service_name=service_name,
            archetype=rules["archetype"],
            center_name=center_name,
            tone_of_voice=tone_of_voice,
            brand_keywords=brand_keywords,
        )

        results.append({
            "appointment_id": c["id"],
            "service_name": service_name,
            "archetype": rules["archetype"],
            "content_type": "post",
            "rationale": f"Servizio {service_name} del {c['giorno']}",
            "material_checklist": rules["checklist"],
            "caption_text": caption,
            "hashtags": hashtags,
        })

    logger.info(f"Piano creato per {len(results)} contenuti")
    return results


def _generate_caption_and_hashtags(
    service_name: str,
    archetype: str,
    center_name: str,
    tone_of_voice: str,
    brand_keywords: list,
) -> tuple[str, list[str]]:
    """Chiama Gemini solo per caption e hashtag."""

    archetype_hint = {
        "before_after": "mostra la trasformazione prima/dopo",
        "editorial": "valorizza il risultato estetico del trattamento",
        "educational": "spiega il trattamento in modo semplice",
        "behind_scenes": "mostra il processo e le mani al lavoro",
        "promo": "invita a prenotare con un'offerta",
    }.get(archetype, "valorizza il trattamento")

    prompt = f"""Sei un esperto di social media per centri estetici italiani.

Scrivi una caption Instagram e una lista di hashtag per questo contenuto:

CENTRO: {center_name}
SERVIZIO: {service_name}
TIPO DI POST: {archetype_hint}
TONO: {tone_of_voice}
PAROLE CHIAVE BRAND: {", ".join(brand_keywords) if brand_keywords else "cura, benessere, bellezza"}

REGOLE CAPTION:
- Prima riga: menziona il nome esatto del servizio "{service_name}"
- Descrivi il risultato specifico di QUESTO servizio (non frasi generiche)
- CTA finale (es. "Prenota ora → link in bio")
- 2-3 emoji pertinenti
- Max 100 parole
- NON usare: "ogni trattamento", "la costanza premia", "risultati visibili"

HASHTAG: 8-10 hashtag specifici per {service_name} (no solo generici).

Rispondi SOLO con JSON valido:
{{
  "caption": "testo caption...",
  "hashtags": ["hashtag1", "hashtag2"]
}}"""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=MODEL_TEXT,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json",
            ),
        )
        data = _parse_json_response(response.text)
        return data.get("caption", ""), data.get("hashtags", [])
    except Exception as e:
        logger.error(f"Errore generazione caption per {service_name}: {e}")
        return f"Scopri il trattamento {service_name} da {center_name}! Prenota ora → link in bio", []


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
    service_name = content_record.get("service_name", "trattamento")
    notes = content_record.get("estetista_notes") or ""
    consent = content_record.get("client_consent", "details_only")

    social_profile = tenant.get("social_profile") or {}
    center_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    style = social_profile.get("style", "minimal")
    tone = social_profile.get("tone_of_voice", "professionale e caldo")
    if isinstance(tone, list):
        tone = ", ".join(tone)
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

    prompt = f"""Sei un designer esperto di contenuti social per centri estetici italiani.

Stai pianificando un contenuto per: {center_name}
Servizio: {service_name}
Tipo di contenuto: {archetype}
Stile brand: {style} — tono {tone}
Colori brand: principale {primary_color}, secondario {secondary_color}
{f"Note estetista: {notes}" if notes else ""}
Regola privacy: {consent_instruction}

Analizza le foto che ti mando e descrivi IN ITALIANO SEMPLICE cosa creerai.
Scrivi come se stessi spiegando a un'amica cosa stai per fare, con entusiasmo.

Struttura la risposta in questi paragrafi:

📐 LAYOUT
(Come sistemerai le foto — es. "Metterò la tua foto al centro...")

🎨 STILE
(Colori, sfondo, effetti — es. "Lo sfondo sarà color crema...")

✍️ TESTO
(Cosa scriverai sopra — es. "In alto metterò il nome del servizio...")

✨ EFFETTO FINALE
(Come apparirà il post finito)

Regole:
- Italiano semplice, NO termini tecnici
- Max 120 parole totali
- Usa SOLO le foto fornite, non inventare elementi non presenti"""

    try:
        client = _get_client()
        contents: list = [prompt] + pil_images

        response = await client.aio.models.generate_content(
            model=MODEL_TEXT,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.7),
        )
        brief = response.text.strip()
        logger.info(f"Brief visivo generato per content {content_record.get('id')}")
        return brief
    except Exception as e:
        logger.error(f"Errore generazione brief: {e}")
        return "Non è stato possibile generare il brief visivo. Riprova."


# ── 3. Generazione immagine ────────────────────────────────────────

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
    service_name = content_record.get("service_name", "trattamento")
    brief = content_record.get("visual_brief_override") or content_record.get("visual_brief") or ""

    social_profile = tenant.get("social_profile") or {}
    center_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    style = social_profile.get("style", "minimal")
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

    archetype_fallback = {
        "before_after": "Split composition BEFORE (left) / AFTER (right). Labels 'PRIMA'/'DOPO' in serif font. Rose gold divider.",
        "editorial": "Elegant editorial post. Photo centered on champagne background. Service name as serif title.",
        "educational": "Informational post. Photo left 60%, text with key benefits right 40%.",
        "behind_scenes": "Warm behind-the-scenes. Natural feel, golden overlay, simple branded text.",
        "promo": "Promotional post. Bold composition. 'Prenota ora' call-to-action.",
    }

    if brief:
        composition = f"VISUAL PLAN — follow this exactly (user approved):\n{brief}"
    else:
        composition = f"COMPOSITION:\n{archetype_fallback.get(archetype, archetype_fallback['editorial'])}"

    prompt = (
        f"You are a luxury beauty brand graphic designer for an Italian aesthetic center.\n\n"
        f"CENTER: {center_name}\n"
        f"SERVICE: {service_name}\n"
        f"STYLE: {style} — luxury, elegant, feminine\n"
        f"BRAND COLORS: primary {primary_color}, secondary {secondary_color}, background champagne #fdf5f0\n"
        f"PRIVACY: {consent_instruction}\n\n"
        f"{composition}\n\n"
        f"BRANDING: Add '{center_name}' in a bottom bar using color {primary_color}, white serif text.\n"
        f"FORMAT: Instagram-ready 1:1 square. Style: Charlotte Tilbury / Dior Beauty luxury.\n\n"
        f"CRITICAL: Use ONLY the photos provided. Do NOT invent or add photos not given to you. "
        f"If only one photo is provided, create a single-photo composition.\n\n"
        f"Create a stunning professional social media post."
    )

    try:
        client = _get_client()
        contents: list = [prompt] + pil_images

        response = await client.aio.models.generate_content(
            model=MODEL_IMAGE,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
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
