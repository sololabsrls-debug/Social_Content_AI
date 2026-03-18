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

# Vocabolario grafico condiviso — elementi decorativi che Gemini può usare
_GRAPHIC_VOCABULARY = """
GRAPHIC VOCABULARY — use sparingly, maximum 1-2 elements total:
• Thin ornamental line: a single hair-thin horizontal line (─────◇─────) in header or footer
• Vignette: very subtle dark gradient fading from edges — enhances depth without adding clutter
• Polaroid frame: white border (thicker at bottom), gives a clean framed feel
• Color swatch: small filled circle showing nail/dye color — only when relevant
• Illustrated icon: a minimal line-drawn beauty element in one corner only
• Translucent strip: semi-transparent bar (dark or brand color) at top or bottom edge for text

LESS IS MORE — one element done with precision is better than three fighting for attention.
Empty space is not a problem. Restraint is sophistication.
"""

# Concetti visivi per archetype — specifici e distinti, non generici
_ARCHETYPE_CREATIVE_GOALS: dict = {
    "before_after": (
        "Communicate transformation — the before/after contrast must hit in 1 second. "
        "Make the difference feel dramatic and earned."
    ),
    "editorial": (
        "Create a beauty image someone would save and share — magazine quality, aspirational. "
        "The result IS the message. Less text, more impact."
    ),
    "educational": (
        "Inform warmly — readable in 3 seconds on a phone. "
        "Inviting and clear, never clinical. Make someone think 'I want that.'"
    ),
    "behind_scenes": (
        "Show the real work, the real hands, the real moment. "
        "Authenticity is the message — resist over-designing."
    ),
    "promo": (
        "Stop the scroll and drive a booking. Bold, direct, zero ambiguity. "
        "Confident energy — not desperate, not cheap."
    ),
}

_ARCHETYPE_INSPIRATIONS: dict = {
    "before_after": [
        "TRANSFORMATION header: thin top bar in primary color with ornamental lines + 'PRIMA · DOPO' "
        "in spaced small caps. BEFORE photo left half and AFTER photo right half, each photo fills "
        "its half of the frame edge-to-edge. Thin ornamental divider at center (❖). "
        "Footer bar with center name + sparkle ornaments ✦",

        "SPLIT FULL-BLEED: BEFORE photo left half, AFTER photo right half, both nearly full-height. "
        "Thin ornamental divider at center (❖ or ◇). Labels 'PRIMA' / 'DOPO' in bold italic serif "
        "overlaid on photos. Narrow brand-color strip at bottom with center name.",

        "POLAROID DIPTYCH: Both photos as polaroid cards (white border, rotated in opposite directions), "
        "each polaroid large (minimum 40% of frame width) — they nearly fill the frame, touching edges. "
        "Hand-lettered 'PRIMA' / 'DOPO' in polaroid bottom margin. Brand background only visible in narrow gaps.",

        "REVEAL CONCEPT: AFTER photo full-frame with a vertical strip of BEFORE visible on the left "
        "edge (like a page being turned). 'PRIMA ↔ DOPO' label. Monogram circle top corner.",
    ],
    "editorial": [
        "POLAROID ON COLOR: Photo as a large polaroid card (white border, thicker bottom, rotated 3-4°) "
        "— the polaroid itself covers 75-80% of the frame. Only a thin sliver of brand background "
        "is visible around it. 'IL RISULTATO' small caps overlaid top-left corner. "
        "Service name in large italic serif in polaroid bottom margin. Decorative dots · on background corners.",

        "FULL-BLEED VIGNETTE: Photo fills nearly the entire frame with dark vignette at edges. "
        "Service name as LARGE italic serif overlay, partially transparent. Center name small bottom. "
        "Sparkle accents ✦ scattered subtly at corners.",

        "LUXURY COLOR CARD: If service involves color (nail, hair, dye) — photo left 55%, "
        "dark brand-color panel right 45%. At top: 'SHADE' in spaced small caps. "
        "Circular color swatch. Color/shade name in large italic serif. Center name + thin line at bottom.",

        "ASYMMETRIC TYPOGRAPHY: Photo occupies right 60%, slightly cropped. Left strip: brand background. "
        "Service name rotated 90° reading bottom-to-top in brand primary, very large. "
        "Thin ornamental line between strip and photo. Monogram circle at top.",
    ],
    "educational": [
        "ILLUSTRATED CONCEPT: Service-specific illustrated icon (drawn eye for lashes, nail shape for "
        "nails, leaf for organic treatments) as central graphic on gradient background. "
        "Service name in large serif below. Tagline. CTA bar at bottom. Can work without photos.",

        "SPLIT PANEL INFO: Photo left 55%, brand-color info panel right. Panel has: service name "
        "bold top, ornamental divider ─◇─, 3 benefit points with sparkle bullets ✦, center name bottom.",

        "NUMBERED STEPS: Photo at top 40% full-width. Below on brand background: "
        "3 numbered steps or facts, each preceded by a circle number in accent color. "
        "Ornamental header line with diamond divider.",

        "ICON GRID: No large photo needed — 4-quadrant grid, each quadrant has a small illustrated "
        "icon + 1-line benefit. Center photo or color as background. Service name as title. Brand colors.",
    ],
    "behind_scenes": [
        "FULL-BLEED + STARS: Photo nearly fills the entire frame. Dark vignette fading from edges. "
        "Small sparkle star accents ✦ ✧ scattered at corners. Service name SMALL CAPS "
        "in a thin header bar. Center name in footer bar. Minimal, luxury.",

        "TILTED POLAROID: Single photo as a large polaroid card (white border, rotated 2-3°) — "
        "polaroid covers 75%+ of the frame, only thin brand-color corners visible. "
        "Service name handwritten-style in polaroid bottom margin. Center name in tiny font below. "
        "2-3 small decorative dots in the exposed background corners only.",

        "STORY TAG AUTHENTIC: Photo full-frame, authentic. Top: pill-shaped location tag "
        "in primary color with center name. Bottom: thin translucent strip with service name. "
        "That's it — raw and real.",

        "CINEMATIC FRAME: Photo with thin inner border/frame in accent color (like a cinema frame). "
        "Top bar: center name in small caps. Bottom: service name italic. Letterbox feel.",
    ],
    "promo": [
        "ILLUSTRATED PROMO: No client photo needed — illustrated beauty icon on gradient background "
        "(primary → secondary color). Service name LARGE serif center. Italic tagline. "
        "Solid CTA bar at bottom in dark primary. Works even without a photo.",

        "BOLD SPLIT: Photo left 50%. Right 50%: solid primary color block. "
        "Service name large + bold serif. Thin ornamental divider line. "
        "'Prenota ora' with accent underline. Center name small + footer ornament ─◇─.",

        "FULL-BLEED CTA: Photo fills frame. Dark semi-transparent overlay on bottom third. "
        "Service name large italic white. 'Prenota ora →' in accent color below. "
        "Center name small. Sparkle accents ✦ top corners.",

        "HEADER FOCUS: Large decorative header bar in primary with ornamental scrollwork. "
        "Photo below (full width, 55%). Footer bar with center name + CTA.",
    ],
}

# Direzione degli elementi grafici per stile visivo del brand
_STYLE_GRAPHIC_DIRECTION: dict = {
    "minimal": (
        "BRAND GRAPHIC STYLE: Clean restraint. Use thin lines, generous whitespace, "
        "at most 1 delicate ornamental element. Typography does the heavy lifting. "
        "The polaroid frame or thin ornamental divider works well. Nothing garish."
    ),
    "luxury": (
        "BRAND GRAPHIC STYLE: Refined opulence. Ornamental dividers with diamond symbols (◇ ❖), "
        "sparkle accents (✦ ✧), wide letter-spacing on all caps text, monogram circle. "
        "Gold/accent color details. Think premium beauty editorial — Vogue Italia level."
    ),
    "naturale": (
        "BRAND GRAPHIC STYLE: Organic warmth. Soft illustrated botanical elements, "
        "brushstroke decorative accents, warm vignette on photos. "
        "Handwritten-feel typography. Earthy and genuine, not corporate."
    ),
    "colorato": (
        "BRAND GRAPHIC STYLE: Bold and joyful. Overlapping colored shapes, strong color blocks, "
        "energetic composition. Can be more graphic and less restrained than other styles. "
        "The illustrated icon concept works especially well here."
    ),
    "moderno": (
        "BRAND GRAPHIC STYLE: Contemporary sharpness. Bold geometric elements — "
        "diagonal lines, rectangle blocks, strong typographic presence. "
        "High contrast between primary and background. Confident and forward."
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
    # service_name viene dal join service:services(name) — NON dal campo diretto
    service_data = content_record.get("service") or {}
    service_name = service_data.get("name") or content_record.get("service_name") or ""
    service_desc = service_data.get("descrizione_breve") or ""
    service_benefits = service_data.get("benefici") or ""
    notes = content_record.get("estetista_notes") or ""
    consent = content_record.get("client_consent", "details_only")

    brand_system_prompt = _build_brand_system_prompt(tenant)
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
    brief = content_record.get("visual_brief_override") or content_record.get("visual_brief") or ""

    brand_system_prompt = _build_brand_system_prompt(tenant)
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

    # Concetti visivi per archetype
    inspirations = _ARCHETYPE_INSPIRATIONS.get(archetype, _ARCHETYPE_INSPIRATIONS["editorial"])
    insp_str = "\n\n".join(f"  CONCEPT {i+1}: {c}" for i, c in enumerate(inspirations))
    creative_goal = _ARCHETYPE_CREATIVE_GOALS.get(archetype, "Create a beautiful, on-brand Instagram post.")

    if brief:
        composition = f"APPROVED VISUAL BRIEF — follow this precisely, the user already approved it:\n{brief}"
    else:
        composition = (
            f"CREATIVE GOAL: {creative_goal}\n\n"
            f"VISUAL CONCEPTS — pick one and execute it with precision, or combine/adapt:\n"
            f"{insp_str}\n\n"
            f"{graphic_direction}\n\n"
            f"{_GRAPHIC_VOCABULARY}"
        )

    # Contesto servizio opzionale per arricchire la generazione
    service_context_en = ""
    if service_desc:
        service_context_en += f"Service description: {service_desc}\n"
    if service_benefits:
        service_context_en += f"Key benefits: {service_benefits}\n"

    # Testi nella grafica — solo nome servizio obbligatorio, nome centro opzionale
    required_texts = f"'{service_name}'" if service_name else ""

    prompt = (
        f"You are a creative director at a top Italian beauty brand agency. "
        f"Create a clean, elegant Instagram post — refined and delicate, never cluttered.\n\n"

        f"BRAND: {center_name}\n"
        f"SERVICE: {service_name or '(see photos)'}\n"
        f"{service_context_en}"
        f"PRIVACY: {consent_instruction}\n\n"

        f"BRAND IDENTITY:\n"
        f"  Feel: {style_feel}\n"
        f"  Photo treatment: {photo_treatment}\n"
        f"  Typography: {font_style}\n"
        f"  Colors (use EXACTLY — no substitutions):\n"
        f"    Primary {primary_color} · Secondary {secondary_color} · Accent {accent_color} · Background {bg_color}\n\n"

        f"{composition}\n\n"

        f"PHOTO INTEGRITY — non-negotiable:\n"
        f"The photos are REAL treatment results taken by the stylist — authentic, unaltered shots. "
        f"DO NOT redraw, reinterpret, reimagine, or stylize the photo content. "
        f"Do not change nail colors, lash shape, skin tone, eye shape, or ANY visible detail. "
        f"Your task is COMPOSITING: add text and graphic elements ON TOP of the exact unmodified photo. "
        f"Think Photoshop layers — the photo layer stays untouched beneath everything else.\n\n"

        f"PHOTO AS CANVAS — non-negotiable:\n"
        f"DEFAULT layout: the photo fills the ENTIRE 1:1 frame edge-to-edge. "
        f"All text and graphic elements are overlaid on top. "
        f"The ONLY layouts where you may reduce the photo area:\n"
        f"  • Split panel / before-after: photo fills one full side (minimum 50% of frame)\n"
        f"  • Polaroid frame: the polaroid card must be LARGE — at least 70% of frame width\n"
        f"ABSOLUTELY FORBIDDEN: a small photo floating in a large solid brand-color background. "
        f"If brand background color dominates the frame → you failed. "
        f"Photo = the canvas; graphic elements = small accents on top.\n\n"

        f"TREATMENT VISIBILITY — non-negotiable:\n"
        f"The treatment result (nails, lashes, brows, skin) is the main subject — it must be "
        f"100% visible and unobstructed. NEVER place text, labels, or graphic elements over "
        f"the treatment area. Place text only in: thin strips at the very top or bottom edge, "
        f"empty corners of the photo, or on a narrow translucent bar that sits outside the subject.\n\n"

        f"DESIGN PRINCIPLE — clean and delicate:\n"
        f"Use maximum 1-2 graphic elements. Prefer thin lines over thick borders. "
        f"Prefer light, delicate overlays over solid color blocks. "
        f"Prefer generous empty space over decorative clutter. "
        f"Typography should be small and refined, not dominant. "
        f"The photo tells the story — design only whispers context.\n\n"

        f"ABSOLUTE RULES:\n"
        f"- Use ONLY the photos provided — never generate, invent, or replace them\n"
        f"- Brand colors EXACTLY as specified — zero deviation\n"
        + (f"- Include this text on the graphic: {required_texts}\n" if required_texts else "")
        + f"- 1:1 square format, publication-ready\n\n"

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
