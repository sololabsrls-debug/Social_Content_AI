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


# ── Brand system prompt ────────────────────────────────────────────

def _build_brand_system_prompt(tenant: dict) -> str:
    """
    Assembla il system instruction completo del brand da passare a TUTTE
    le chiamate Gemini. Così ogni output (caption, istruzioni foto, brief,
    immagine) rispetta automaticamente voce, stile e regole del centro.
    """
    sp = tenant.get("social_profile") or {}
    name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    bio = (tenant.get("bio") or "").strip()

    # Identità
    tagline = sp.get("tagline") or ""
    city = sp.get("city") or ""
    positioning = sp.get("price_positioning") or "mid-range"
    usp = sp.get("unique_selling_point") or ""

    # Audience
    target = sp.get("target_description") or "Donne 25-45 anni che si prendono cura di sé"

    # Voce
    tone = sp.get("tone_of_voice") or "caldo e professionale"
    if isinstance(tone, list):
        tone = ", ".join(tone)
    comm_style = sp.get("communication_style") or "informale, dai del tu"
    emoji_usage = sp.get("emoji_usage") or "moderato (2-3 per post)"

    avoid = sp.get("avoid_words") or ["ogni trattamento", "la costanza premia", "risultati visibili"]
    avoid_str = ", ".join(f'"{w}"' for w in avoid)

    signatures = sp.get("signature_phrases") or []
    sig_block = "\n".join(f"  • {p}" for p in signatures) if signatures else "  (nessuna frase impostata)"

    # Contenuto
    pillars = sp.get("content_pillars") or [
        "Risultati dei trattamenti", "Educazione beauty", "Behind the scenes", "Promozioni"
    ]
    pillars_str = "\n".join(f"  • {p}" for p in pillars)

    brand_hashtags = sp.get("brand_hashtags") or []
    hashtags_str = " ".join(brand_hashtags) if brand_hashtags else "(nessuno impostato)"

    # Visivo
    visual_style = sp.get("style") or "minimal e professionale"
    primary_color = tenant.get("theme_primary_color") or "#6b2d4e"
    secondary_color = tenant.get("theme_secondary_color") or "#c9a0b4"

    lines = [
        f'Sei il social media manager esclusivo di "{name}".',
        "La tua missione: creare contenuti autentici che sembrino scritti da una persona vera, non da un robot.",
        "",
        "━━━ IDENTITÀ ━━━",
        f"Centro: {name}",
    ]
    if city:
        lines.append(f"Città: {city}")
    if tagline:
        lines.append(f'Tagline: "{tagline}"')
    if bio:
        lines.append(f"Presentazione: {bio}")
    lines.append(f"Posizionamento: {positioning}")
    if usp:
        lines.append(f"Punto di forza unico: {usp}")
    lines += [
        "",
        "━━━ CLIENTELA TARGET ━━━",
        f"  {target}",
        "",
        "━━━ VOCE DEL BRAND ━━━",
        f"Tono: {tone}",
        f"Stile comunicativo: {comm_style}",
        f"Emoji: {emoji_usage}",
        "",
        "Frasi tipiche del brand (usale come ispirazione, non copiare letteralmente):",
        sig_block,
        "",
        f"NON usare MAI queste parole o frasi: {avoid_str}",
        'NON usare mai frasi generiche come: "ogni trattamento è unico", "la cura parte da te", "risultati che parlano da soli", "prendersi cura di sé"',
        "",
        "━━━ PILASTRI EDITORIALI ━━━",
        pillars_str,
        "",
        "━━━ HASHTAG FISSI DEL BRAND (includili sempre) ━━━",
        f"  {hashtags_str}",
        "",
        "━━━ STILE VISIVO ━━━",
        f"Stile grafico: {visual_style}",
        f"Colori brand: {primary_color} (primario), {secondary_color} (secondario)",
    ]

    return "\n".join(lines)


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
        })

    if not candidati:
        logger.info("Nessun appuntamento candidato per questa settimana")
        return []

    # Limita al content_frequency
    candidati = candidati[:content_frequency]

    # Assembla il system prompt del brand una volta sola
    brand_system_prompt = _build_brand_system_prompt(tenant)

    # Per ogni candidato: Python schema base, Gemini personalizza istruzioni + caption
    results = []
    for c in candidati:
        service_name = c["servizio"]
        rules = _get_service_rules(service_name)

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
        data = _parse_json_response(response.text)
        items_map = {item["id"]: item["instructions"] for item in data.get("instructions", [])}

        # Merge: struttura base Python + istruzioni personalizzate Gemini
        result = []
        for base_item in base_checklist:
            item_copy = dict(base_item)
            if base_item["id"] in items_map and items_map[base_item["id"]].strip():
                item_copy["instructions"] = items_map[base_item["id"]]
            result.append(item_copy)
        return result
    except Exception as e:
        logger.error(f"Errore personalizzazione checklist per {service_name}: {e}")
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

    brand_system_prompt = _build_brand_system_prompt(tenant)
    social_profile = tenant.get("social_profile") or {}
    center_name = tenant.get("display_name") or tenant.get("name", "Centro Estetico")
    style = social_profile.get("style", "minimal")
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

    prompt = f"""Stai pianificando un contenuto social per: {center_name}
Servizio: {service_name}
Tipo di contenuto: {archetype}
Stile grafico: {style} — colori brand: {primary_color} (primario), {secondary_color} (secondario)
{f"Note estetista: {notes}" if notes else ""}
Regola privacy: {consent_instruction}

Analizza le foto che ti mando e descrivi IN ITALIANO SEMPLICE cosa creerai.
Scrivi come se stessi spiegando a un'amica cosa stai per fare, con entusiasmo.
Rispetta lo stile e il tono del brand descritto nel system instruction.

Struttura la risposta ESATTAMENTE così (usa le emoji come titoli):

📐 LAYOUT
(Come sistemerai le foto — es. "Metterò la tua foto al centro...")

🎨 STILE
(Colori, sfondo, effetti — coerenti col brand)

✍️ TESTO
(Cosa scriverai sopra — coerente col tono del brand)

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

    brand_system_prompt = _build_brand_system_prompt(tenant)
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
