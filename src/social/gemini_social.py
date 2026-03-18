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
}

# Fallback per servizi non riconosciuti
_DEFAULT_RULES_V2: dict = {
    "rotation_weights": {"editorial": 40, "educational": 30, "behind_scenes": 20, "before_after": 10},
    "archetypes": {
        "editorial": [
            {"id": "result", "label": "Foto del risultato", "required": True, "instructions": ""},
        ],
        "educational": [
            {"id": "treatment", "label": "Foto del trattamento", "required": True, "instructions": ""},
        ],
        "behind_scenes": [
            {"id": "process", "label": "Foto durante il trattamento", "required": True, "instructions": ""},
        ],
        "before_after": [
            {"id": "before", "label": "Foto PRIMA", "required": True, "instructions": ""},
            {"id": "after",  "label": "Foto DOPO",  "required": True, "instructions": ""},
        ],
    },
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

    # Limita al content_frequency
    candidati = candidati[:content_frequency]

    # Assembla il system prompt del brand una volta sola
    brand_system_prompt = _build_brand_system_prompt(tenant)

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


# ── Creative briefs per archetype ─────────────────────────────────
#
# Invece di template fissi da seguire, Gemini riceve:
#   1. Un OBIETTIVO creativo per l'archetype (cosa deve comunicare)
#   2. ISPIRAZIONE con possibili approcci (non istruzioni rigide)
#   3. VINCOLI di brand (colori, font, privacy) — questi sì rigidi
#
# Gemini ha libertà creativa su layout, composizione, elementi grafici,
# posizione testi. I post usciranno visivamente diversi tra loro.

# Obiettivo creativo per archetype — cosa deve comunicare il post
_ARCHETYPE_CREATIVE_GOALS: dict = {
    "before_after": (
        "Show the dramatic transformation this service creates. "
        "The viewer must grasp the before/after contrast instantly. "
        "Make the difference unmissable — this is the strongest social proof format."
    ),
    "editorial": (
        "Create an aspirational, magazine-quality beauty image. "
        "Elegant, curated, desirable. The kind of post someone saves and shares. "
        "The result is the hero — let it speak with minimal clutter."
    ),
    "educational": (
        "Design a clear, warm infographic that explains the service's main benefits. "
        "Readable on a phone in 3 seconds. Informative but inviting — not clinical. "
        "Make someone think 'I didn't know that, I should book this.'"
    ),
    "behind_scenes": (
        "Capture an authentic human moment of the beauty work in progress. "
        "Raw, real, trustworthy. Minimal design — the photo is the hero, not the graphics. "
        "Should feel like a genuine peek behind the curtain, not a staged shot."
    ),
    "promo": (
        "Create a high-impact post that makes someone stop scrolling and book. "
        "Bold, direct, clear CTA. Confident and warm — not pushy or cheap-looking."
    ),
}

# Ispirazione per composizione — Gemini può scegliere, adattare, combinare o inventare
_ARCHETYPE_INSPIRATIONS: dict = {
    "before_after": [
        "Classic 50/50 vertical split with a clean dividing line",
        "Diagonal composition — BEFORE fills one triangle, AFTER the other",
        "AFTER full-frame with BEFORE as a smaller polaroid inset in one corner",
        "Triptych: BEFORE | a detail/treatment step | AFTER in three panels",
        "BEFORE fades into AFTER using a horizontal reveal/wipe effect",
    ],
    "editorial": [
        "Photo centered dominant, elegant title below on brand background",
        "Asymmetric: photo right side, large typographic title in left column",
        "Full-bleed photo with color gradient overlay and text at bottom",
        "Photo with a decorative color frame or border in brand accent color",
        "Photo slightly cropped/zoomed, generous negative space filled with brand color",
        "Close-up detail shot with service name as large background watermark text",
    ],
    "educational": [
        "Photo left, clean info panel right with 3-4 benefit bullet points",
        "Photo top, title + benefit grid (2×2 or 3 columns) below",
        "Large bold number or icon as graphic anchor, photo and facts alongside",
        "Step-by-step horizontal strip: icon — text — icon — text",
        "Circular photo inset in a graphic layout with text surrounding it",
    ],
    "behind_scenes": [
        "Nearly full-bleed photo, only service name and center name in a corner",
        "Polaroid frame with white border and slight tilt — raw and organic",
        "Instagram story-style with a location tag pill overlay at top",
        "Two candid moments side by side as a diptych",
        "Photo with a subtle warm color wash and handwritten-style caption",
    ],
    "promo": [
        "50/50 split: photo left, bold brand-color block right with service + CTA",
        "Photo fills the frame, centered semi-transparent text box overlay",
        "Photo top 60%, bold brand-color bottom bar with service name + 'Prenota ora'",
        "Full-bleed photo with large bold text in brand color over a partial overlay",
        "Circular photo in center, graphic shapes surrounding it, CTA prominent",
    ],
}

# Direzione degli elementi grafici per stile visivo del brand
_STYLE_GRAPHIC_DIRECTION: dict = {
    "minimal": (
        "GRAPHIC STYLE: Extreme restraint — generous whitespace, thin lines only. "
        "No decorative shapes. Let the photo and typography breathe. Less is always more."
    ),
    "luxury": (
        "GRAPHIC STYLE: Refined luxury — optional hairline decorative lines in the accent color, "
        "wide letter-spacing on typography. Subtle, never garish. Think Vogue, not QVC."
    ),
    "naturale": (
        "GRAPHIC STYLE: Organic warmth — soft botanical or brushstroke decorative elements at low opacity, "
        "warm color palette. Feels handcrafted and genuine, not corporate."
    ),
    "colorato": (
        "GRAPHIC STYLE: Bold and joyful — overlapping geometric shapes (circles, arcs) in brand colors "
        "as background accents. Strong color blocks. Energetic without being chaotic."
    ),
    "moderno": (
        "GRAPHIC STYLE: Contemporary and sharp — bold geometric elements, diagonal lines or blocks "
        "in brand colors. High contrast. Confident typographic presence."
    ),
}


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

    prompt = f"""Stai per creare un post per: {center_name}
Servizio: {service_name}
Tipo di post: {archetype}
{f"Note dell'estetista: {notes}" if notes else ""}
Regola privacy: {consent_instruction}

PALETTE BRAND (usa ESATTAMENTE questi colori, descrivendoli in italiano):
- Primario: {primary_color}
- Secondario: {secondary_color}
- Accento: {accent_color}
- Sfondo: {bg_color}

STILE FOTO: {photo_style_desc}
TIPOGRAFIA: {typo_desc}

OBIETTIVO DEL POST: {archetype_goal_it}

Guarda le foto e immagina liberamente il post più bello e d'impatto che puoi creare
per questo servizio con queste foto. Hai libertà creativa completa su layout e composizione.

Descrivi IN ITALIANO SEMPLICE la tua idea — come se spiegassi a un'amica cosa stai
per fare. Struttura così:

📐 IDEA
(In 1-2 frasi: che tipo di composizione hai scelto e perché funziona per questo contenuto)

🎨 STILE
(Colori usati — descritti in parole, non codici. Trattamento foto. Atmosfera)

✍️ TESTO
(Cosa scrivi, dove lo metti, con che font/stile)

✨ EFFETTO
(Che sensazione trasmetterà? Cosa farà fermare chi scorre il feed?)

Regole:
- Italiano semplice, ZERO termini tecnici ("overlay", "compositing", "gradient" non esistono)
- Descrivi i colori con parole normali (es. "viola profondo", "rosa antico", "crema")
- Max 150 parole
- Usa SOLO le foto fornite — non inventare elementi assenti"""

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

    sp = social_profile  # alias locale
    visual_style = sp.get("visual_style") or sp.get("style") or "minimal"
    photo_style = sp.get("photo_style") or "bright_natural"
    typo_style = sp.get("typography_style") or "serif_elegant"
    accent_color = sp.get("accent_color") or secondary_color
    bg_color = sp.get("background_color") or "#fdf5f0"

    photo_treatment = {
        "bright_natural": "bright, natural lighting — warm window light feel, vivid and clean colors",
        "warm_moody":     "warm moody treatment — amber tones, soft shadows, intimate and cozy feel",
        "clean_white":    "clean clinical look — flat even lighting, neutral white balance, ultra clean",
        "dark_luxury":    "dark luxury feel — deep rich tones, strong contrasts, premium and dramatic",
    }.get(photo_style, "bright, natural lighting")

    font_style = {
        "serif_elegant": "elegant serif font (Playfair Display or similar)",
        "sans_modern":   "modern clean sans-serif (Inter or Helvetica)",
        "mixed":         "serif for titles, sans-serif for supporting text",
    }.get(typo_style, "elegant serif font")

    style_feel = {
        "minimal":   "minimal, clean and restrained — lots of breathing space, nothing unnecessary",
        "luxury":    "opulent luxury — rich textures, refined details, premium editorial feel",
        "naturale":  "natural and organic — earthy palette, soft edges, botanical warmth",
        "colorato":  "vibrant and joyful — bold colors, energetic, warm and inviting",
        "moderno":   "modern and bold — geometric elements, strong type, contemporary edge",
    }.get((visual_style or "minimal").lower().split()[0], "clean and professional")

    graphic_direction = _STYLE_GRAPHIC_DIRECTION.get(
        (visual_style or "minimal").lower().split()[0], _STYLE_GRAPHIC_DIRECTION["minimal"]
    )

    # Ispirazione composizione — Gemini sceglie o inventa liberamente
    inspirations = _ARCHETYPE_INSPIRATIONS.get(archetype, _ARCHETYPE_INSPIRATIONS["editorial"])
    insp_str = "\n".join(f"  • {i}" for i in inspirations)
    creative_goal = _ARCHETYPE_CREATIVE_GOALS.get(archetype, "Create a beautiful, on-brand Instagram post.")

    if brief:
        composition = f"APPROVED VISUAL BRIEF — follow this precisely, the user already approved it:\n{brief}"
    else:
        composition = (
            f"CREATIVE GOAL: {creative_goal}\n\n"
            f"LAYOUT INSPIRATION — choose one, combine, adapt, or create something entirely different:\n"
            f"{insp_str}\n\n"
            f"CREATIVE FREEDOM: You are NOT locked to these options. "
            f"Design the composition that best serves the goal and the photos you see. "
            f"Surprise creatively while staying on-brand.\n\n"
            f"{graphic_direction}"
        )

    prompt = (
        f"You are a creative graphic designer making an Instagram post for an Italian beauty center.\n\n"
        f"CENTER: {center_name}\n"
        f"SERVICE: {service_name}\n"
        f"PHOTO TREATMENT: {photo_treatment}\n"
        f"TYPOGRAPHY: {font_style}\n"
        f"BRAND COLORS (use EXACTLY these — never substitute):\n"
        f"  Primary: {primary_color} · Secondary: {secondary_color} "
        f"· Accent: {accent_color} · Background: {bg_color}\n"
        f"OVERALL FEEL: {style_feel}\n"
        f"PRIVACY: {consent_instruction}\n\n"
        f"{composition}\n\n"
        f"NON-NEGOTIABLE RULES:\n"
        f"- Use ONLY the photos given. Never add or invent photos.\n"
        f"- Brand colors must match EXACTLY.\n"
        f"- Must include center name '{center_name}' and service name '{service_name}'.\n"
        f"- Result must be publication-ready, 1:1 square format (1080×1080px).\n\n"
        f"Create the image now."
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
