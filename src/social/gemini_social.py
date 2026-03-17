"""
Logica AI del modulo social — tutte le chiamate Gemini.

Tre funzioni principali:
1. select_and_plan_week()   → sceglie appuntamenti e genera piano + istruzioni + testo
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
    """Estrae JSON dalla risposta Gemini anche se wrappato in markdown."""
    text = text.strip()
    # Rimuove ```json ... ``` se presente
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    return json.loads(text)


async def _download_image(url: str) -> Image.Image:
    """Scarica immagine da URL e restituisce PIL Image."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")


def _image_to_pil(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _create_story_version(feed_image: Image.Image) -> bytes:
    """
    Crea versione 9:16 story partendo dall'immagine feed 1:1.
    Aggiunge bande champagne in alto e basso con gradient.
    """
    w, h = feed_image.size
    story_h = int(w * 16 / 9)
    background_color = (253, 245, 240)  # champagne

    story = Image.new("RGB", (w, story_h), background_color)
    y_offset = (story_h - h) // 2
    story.paste(feed_image, (0, y_offset))

    out = io.BytesIO()
    story.save(out, format="JPEG", quality=90)
    return out.getvalue()


def _extract_image_from_response(response) -> Optional[bytes]:
    """Estrae bytes immagine dalla risposta Gemini."""
    for part in response.candidates[0].content.parts:
        if part.inline_data and "image" in part.inline_data.mime_type:
            return part.inline_data.data
    return None


# ── 1. Selezione e pianificazione settimanale ──────────────────────

def select_and_plan_week(
    appointments: list[dict],
    tenant: dict,
    week_start_str: str,
    week_end_str: str,
    existing_appointment_ids: list[str],
) -> list[dict]:
    """
    Gemini analizza gli appuntamenti e crea il piano editoriale della settimana.

    Regole fisse applicate prima di passare a Gemini:
    - Max 4 post/settimana (o content_frequency del tenant)
    - Escludi appuntamenti già pianificati questa settimana
    - Escludi servizi non fotografabili

    Gemini poi decide:
    - Quali scegliere tra i candidati
    - Archetype più adatto per ognuno
    - Giorno di pubblicazione consigliato
    - Istruzioni materiale in italiano semplice
    - Caption e hashtag

    Ritorna lista di dict con il piano per ogni contenuto.
    """
    social_profile = tenant.get("social_profile") or {}
    content_frequency = social_profile.get("content_frequency", 3)
    tone_of_voice = social_profile.get("tone_of_voice", "caldo e professionale")
    style = social_profile.get("style", "minimal")
    brand_keywords = social_profile.get("brand_keywords", [])
    center_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    bio = tenant.get("bio") or ""

    # Regola fissa: servizi non fotografabili da escludere
    NON_FOTOGRAFABILI = [
        "ceretta", "epilazione ascelle", "epilazione inguine", "epilazione bikini",
        "massaggio", "drenante", "pressoterapia", "cavitazione"
    ]

    # Prepara sommario appuntamenti
    appt_summary = []
    for a in appointments:
        appt_id = a.get("id")
        if appt_id in existing_appointment_ids:
            continue
        service = a.get("service") or {}
        service_name = service.get("name", "")

        # Applica regola fissa: salta servizi non fotografabili
        is_non_foto = any(k.lower() in service_name.lower() for k in NON_FOTOGRAFABILI)
        if is_non_foto:
            continue

        appt_summary.append({
            "id": appt_id,
            "giorno": a.get("start_at", "")[:10],
            "servizio": service_name,
            "descrizione": service.get("descrizione_breve") or "",
            "benefici": service.get("benefici") or [],
            "prodotti": service.get("prodotti_utilizzati") or [],
            "staff": (a.get("staff") or {}).get("name", ""),
        })

    if not appt_summary:
        logger.info("Nessun appuntamento candidato per questa settimana")
        return []

    prompt = f"""Sei un esperto di social media per centri estetici italiani.

CENTRO: {center_name}
BIO: {bio}
TONO DI VOCE: {tone_of_voice}
STILE: {style}
PAROLE CHIAVE BRAND: {", ".join(brand_keywords) if brand_keywords else "naturale, cura, benessere"}
SETTIMANA: {week_start_str} — {week_end_str}
POST DA PIANIFICARE: massimo {content_frequency}

APPUNTAMENTI DISPONIBILI:
{json.dumps(appt_summary, ensure_ascii=False, indent=2)}

ARCHETIPI DISPONIBILI:
- before_after: per trattamenti con risultato visivo chiaro (laminazione, nail art, microblading)
- editorial: post stile rivista, elegante e aspirazionale
- educational: per spiegare un servizio poco conosciuto
- behind_scenes: dietro le quinte del lavoro quotidiano
- promo: per riempire slot liberi in agenda

REGOLE (già applicate, non ignorare):
1. Seleziona massimo {content_frequency} appuntamenti
2. Varia gli archetipi: non tutti before_after
3. Distribuisci i post in giorni diversi della settimana
4. Privilegia servizi con alto impatto visivo

PER LE ISTRUZIONI MATERIALE — REGOLA FONDAMENTALE:
Scrivi come se stessi spiegando a un'amica NON fotografa cosa fare.
Usa un linguaggio semplice, pratico, rassicurante.
Spiega: cosa fotografare, come tenere il telefono, dove farlo (luce, sfondo),
cosa evitare, quante foto fare. Max 3-4 frasi per item.
NON usare termini tecnici (no: "ISO", "bokeh", "angolazione", "composizione").
Usa invece: "vicino alla finestra", "a una spanna di distanza", "tieni il telefono dritto".

PER LE CAPTION:
- Scrivi in italiano
- Tono coerente con il profilo brand
- Max 150 parole
- Includi sempre una call to action finale

Rispondi SOLO con JSON valido, senza testo prima o dopo:
{{
  "selected": [
    {{
      "appointment_id": "uuid dell'appuntamento",
      "service_name": "nome del servizio",
      "archetype": "before_after",
      "content_type": "post",
      "scheduled_day": "martedi",
      "rationale": "breve motivazione della scelta (1 frase)",
      "material_checklist": [
        {{
          "id": "item_1",
          "label": "Foto PRIMA",
          "instructions": "istruzioni semplici in italiano...",
          "required": true,
          "uploaded_url": null
        }}
      ],
      "caption_text": "testo caption completo...",
      "hashtags": ["hashtag1", "hashtag2"]
    }}
  ]
}}"""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=MODEL_TEXT,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        result = _parse_json_response(response.text)
        selected = result.get("selected", []) if isinstance(result, dict) else result
        logger.info(f"Gemini ha selezionato {len(selected)} contenuti per la settimana {week_start_str}")
        return selected
    except Exception as e:
        logger.error(f"Errore selezione settimanale Gemini: {e}")
        return []


# ── 2. Generazione brief visivo ────────────────────────────────────

async def generate_visual_brief(
    content_record: dict,
    tenant: dict,
) -> str:
    """
    Analizza le foto caricate e descrive IN ITALIANO SEMPLICE
    cosa creerà — mostrato all'estetista PRIMA di generare l'immagine.

    Ritorna testo descrittivo (non JSON).
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
    primary_color = tenant.get("theme_primary_color") or "#6b2d4e"
    secondary_color = tenant.get("theme_secondary_color") or "#c9a0b4"

    # Mappa consenso in istruzione per Gemini
    consent_instruction = {
        "with_face": "Puoi mostrare il viso della cliente.",
        "details_only": "Mostra solo mani, dettagli o zone specifiche, NON il viso.",
        "no_client": "Non mostrare la cliente. Usa solo prodotti e ambiente.",
    }.get(consent, "Mostra solo dettagli, no viso.")

    # Scarica le foto
    pil_images = []
    for url in photos[:3]:  # max 3 foto
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

Struttura la risposta in questi paragrafi (usa le emoji indicate):

📐 LAYOUT
(Come sistemerai le foto — es. "Metterò le tue due foto affiancate...")

🎨 STILE
(Colori, sfondo, effetti — es. "Lo sfondo sarà color crema, aggiungo una cornicetta dorata...")

✍️ TESTO
(Cosa scriverai sopra — es. "In alto metterò la parola PRIMA e DOPO in carattere elegante...")

✨ EFFETTO FINALE
(Come apparirà — es. "Il risultato sarà un post elegante stile rivista...")

Regole:
- Italiano semplice, NO termini tecnici di design
- Max 120 parole totali
- Tono entusiasta ma professionale
- Sii specifico su cosa farai con QUESTE foto"""

    try:
        client = _get_client()
        contents: list = [prompt] + pil_images

        response = client.models.generate_content(
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

    Ritorna (feed_bytes_1x1, story_bytes_9x16).
    La story viene creata da Pillow a partire dal feed (senza costo API extra).
    """
    photos = content_record.get("photos_input") or []
    archetype = content_record.get("archetype", "editorial")
    service_name = content_record.get("service_name", "trattamento")
    caption = content_record.get("caption_text") or ""

    # Usa brief_override se presente, altrimenti brief originale
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

    # Scarica foto
    pil_images = []
    for url in photos[:3]:
        try:
            img = await _download_image(url)
            pil_images.append(img)
        except Exception as e:
            logger.warning(f"Impossibile scaricare foto {url}: {e}")

    # Costruisce prompt immagine
    archetype_instructions = {
        "before_after": (
            "Create a before/after split composition. "
            "First image is BEFORE (left side), second image is AFTER (right side). "
            "Add elegant labels 'PRIMA' and 'DOPO' in serif font. "
            "Add a thin rose gold divider between the photos. "
            "Enhance the AFTER photo: +15% brightness, +10% saturation."
        ),
        "editorial": (
            "Create a high-fashion editorial Instagram post. "
            "Place the photo centrally on a champagne/cream background. "
            "Add the service name as an elegant serif title. "
            "Style: Vogue Beauty meets Italian luxury."
        ),
        "educational": (
            "Create an elegant informational post. "
            "Split the image: photo on left (60%), text area on right (40%). "
            "Add 3-4 key benefit bullet points on the right side. "
            "Clean, readable, professional."
        ),
        "behind_scenes": (
            "Create a warm behind-the-scenes aesthetic post. "
            "Natural, authentic feel. Add a subtle warm golden overlay. "
            "Include a simple branded text element."
        ),
        "promo": (
            "Create a promotional post with clear call-to-action. "
            "Bold, eye-catching composition. "
            "Include a prominent 'Prenota ora' button element."
        ),
    }

    img_instruction = archetype_instructions.get(archetype, archetype_instructions["editorial"])

    prompt = (
        f"You are a luxury beauty brand graphic designer for an Italian aesthetic center.\n\n"
        f"CENTER: {center_name}\n"
        f"SERVICE: {service_name}\n"
        f"STYLE: {style} — luxury, elegant, feminine\n"
        f"BRAND COLORS: primary {primary_color}, secondary {secondary_color}, "
        f"background champagne #fdf5f0\n"
        f"PRIVACY: {consent_instruction}\n\n"
        f"VISUAL PLAN (follow this exactly):\n{brief}\n\n"
        f"COMPOSITION TYPE: {img_instruction}\n\n"
        f"BRANDING REQUIREMENTS:\n"
        f"- Add center name '{center_name}' in a bottom branding bar\n"
        f"- Use primary color {primary_color} for the branding bar\n"
        f"- Text on branding bar in white, elegant serif font\n"
        f"- Overall: Instagram-ready 1:1 square format, 1080x1080px equivalent\n"
        f"- Style inspiration: Charlotte Tilbury, Dior Beauty — luxury accessible\n\n"
        f"Create a stunning, professional social media post."
    )

    try:
        client = _get_client()
        contents: list = [prompt] + pil_images

        response = client.models.generate_content(
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

        # Story 9:16 via Pillow (no costo API extra)
        feed_pil = _image_to_pil(feed_bytes)
        story_bytes = _create_story_version(feed_pil)

        logger.info(f"Immagine generata per content {content_record.get('id')}")
        return feed_bytes, story_bytes

    except Exception as e:
        logger.error(f"Errore generazione immagine: {e}")
        return None, None
