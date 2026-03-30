# AI Graphic Archetype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggiungere un post settimanale completamente generato dall'AI (`ai_graphic`) al piano editoriale, senza che l'estetista carichi foto — l'AI decide concept, testo e grafica, e il post arriva già pronto per approvazione.

**Architecture:** In fondo a `run_weekly_pipeline` si aggiunge 1 post `ai_graphic` generato da `generate_ai_graphic_post()` in `gemini_social.py`. La rotazione tra 4 categorie (tip_beauty, spotlight, stagionale, ispirazione) è tracciata in `tenant.social_profile.last_ai_graphic_category`. Il post è salvato direttamente con `status="draft"` e immagine già generata, pronto per approvazione nel frontend.

**Tech Stack:** Python, FastAPI, Google Gemini 2.5 Flash (testo) + gemini-3-pro-image-preview (immagine), Supabase (PostgreSQL + Storage), Pillow

---

### Task 1: DB Migration — aggiunge `ai_graphic` al CHECK constraint

**Files:**
- Create: `migrations/003_ai_graphic_archetype.sql`

Il CHECK constraint attuale su `archetype` non include `ai_graphic`. Va alterato.

- [ ] **Step 1: Crea il file migration**

```sql
-- migrations/003_ai_graphic_archetype.sql
-- Aggiunge 'ai_graphic' al CHECK constraint archetype in social_content

ALTER TABLE public.social_content
DROP CONSTRAINT IF EXISTS social_content_archetype_check;

ALTER TABLE public.social_content
ADD CONSTRAINT social_content_archetype_check
CHECK (archetype IN (
    'before_after', 'editorial', 'promo',
    'educational', 'behind_scenes', 'retention', 'ai_graphic'
));
```

- [ ] **Step 2: Esegui in Supabase SQL Editor**

Vai su https://fpsrqrzuvjabxocmunhi.supabase.co → SQL Editor → incolla ed esegui.

Verifica con:
```sql
SELECT conname, consrc
FROM pg_constraint
WHERE conrelid = 'public.social_content'::regclass
AND contype = 'c'
AND conname = 'social_content_archetype_check';
```

Atteso: `consrc` contiene `'ai_graphic'`.

- [ ] **Step 3: Commit file migration**

```bash
cd "C:\Users\scate\OneDrive\Desktop\Gestionale_Estetiste\Social_Content_AI\backend"
git add migrations/003_ai_graphic_archetype.sql
git commit -m "feat: aggiungi ai_graphic al CHECK constraint archetype"
```

---

### Task 2: Costanti e logica di rotazione in `gemini_social.py`

**Files:**
- Modify: `src/social/gemini_social.py` (dopo il blocco `_DEFAULT_RULES_V2`)

- [ ] **Step 1: Aggiungi costanti e funzione rotazione**

Aggiungi dopo la riga che chiude `_DEFAULT_RULES_V2` (dopo `}`):

```python
# ── AI Graphic — rotazione categorie settimanali ──────────────────

AI_GRAPHIC_CATEGORIES = ["tip_beauty", "spotlight", "stagionale", "ispirazione"]


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
```

- [ ] **Step 2: Test della logica rotazione (no API)**

Crea file `test_rotation.py` nella root del progetto:

```python
"""Test unitario per _get_next_ai_graphic_category — nessuna API call."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.social.gemini_social import _get_next_ai_graphic_category, AI_GRAPHIC_CATEGORIES

def test_rotation():
    # Nessuna storia → prima categoria
    assert _get_next_ai_graphic_category({}) == "tip_beauty"
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": None}) == "tip_beauty"

    # Avanzamento normale
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": "tip_beauty"}) == "spotlight"
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": "spotlight"}) == "stagionale"
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": "stagionale"}) == "ispirazione"

    # Ciclo completo
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": "ispirazione"}) == "tip_beauty"

    # Valore non valido → ricomincia
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": "xyz"}) == "tip_beauty"

    print("✓ Tutti i test rotazione passati")

if __name__ == "__main__":
    test_rotation()
```

- [ ] **Step 3: Esegui il test**

```bash
cd "C:\Users\scate\OneDrive\Desktop\Gestionale_Estetiste\Social_Content_AI\backend"
python test_rotation.py
```

Atteso: `✓ Tutti i test rotazione passati`

- [ ] **Step 4: Commit**

```bash
git add src/social/gemini_social.py test_rotation.py
git commit -m "feat: rotazione categorie ai_graphic + test unitario"
```

---

### Task 3: Funzione `_pick_spotlight_service` in `gemini_social.py`

**Files:**
- Modify: `src/social/gemini_social.py` (dopo `_get_next_ai_graphic_category`)

- [ ] **Step 1: Aggiungi la funzione**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/social/gemini_social.py
git commit -m "feat: funzione _pick_spotlight_service per rotazione servizi"
```

---

### Task 4: Funzione `_generate_ai_graphic_text` in `gemini_social.py`

**Files:**
- Modify: `src/social/gemini_social.py`

- [ ] **Step 1: Aggiungi la funzione (dopo `_pick_spotlight_service`)**

```python
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
        "ispirazione": (
            f"Crea un post ispirazionale — una citazione o pensiero motivazionale sul benessere e la cura di sé.\n"
            f"La citazione deve essere breve, originale, non banale.\n"
            f"CONCEPT: la citazione stessa (max 20 parole).\n"
            f"CAPTION: 1-2 frasi che collegano il pensiero al brand {center_name}."
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
```

- [ ] **Step 2: Commit**

```bash
git add src/social/gemini_social.py
git commit -m "feat: _generate_ai_graphic_text — testo e caption per le 4 categorie"
```

---

### Task 5: Funzione `_generate_ai_graphic_image` in `gemini_social.py`

**Files:**
- Modify: `src/social/gemini_social.py`

- [ ] **Step 1: Aggiungi la funzione (dopo `_generate_ai_graphic_text`)**

```python
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
        "ispirazione": (
            f"GRAPHIC TYPE: Inspirational quote post\n"
            f"QUOTE (Italian): {concept}\n\n"
            f"LAYOUT: Minimal quote graphic. Quote text centered, large, {typo_style}. "
            f"Soft elegant background ({bg_color} or soft gradient of {primary_color}). "
            f"Minimal decorative element (thin line or small flourish). "
            f"Center name '{center_name}' as small signature at the bottom."
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
```

- [ ] **Step 2: Commit**

```bash
git add src/social/gemini_social.py
git commit -m "feat: _generate_ai_graphic_image — grafica Canva-style senza foto reali"
```

---

### Task 6: Funzione pubblica `generate_ai_graphic_post` in `gemini_social.py`

**Files:**
- Modify: `src/social/gemini_social.py`

- [ ] **Step 1: Aggiungi la funzione orchestratrice (dopo `_generate_ai_graphic_image`)**

```python
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
    brand_system_prompt = _build_brand_system_prompt(tenant)
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
        "ai_graphic_category": category,          # usato dalla pipeline per aggiornare rotazione
        "feed_bytes": feed_bytes,                 # bytes, non salvati su DB
        "story_bytes": story_bytes,               # bytes, non salvati su DB
        "service_id": spotlight_service.get("id") if spotlight_service else None,
    }
```

- [ ] **Step 2: Aggiorna l'import nella sezione import di `content_pipeline.py`**

Aggiungi `generate_ai_graphic_post` all'import esistente da `gemini_social`:

```python
from src.social.gemini_social import (
    select_and_plan_week,
    generate_visual_brief,
    generate_image,
    generate_image_variants,
    generate_ai_graphic_post,          # ← aggiunto
)
```

- [ ] **Step 3: Commit**

```bash
git add src/social/gemini_social.py src/social/content_pipeline.py
git commit -m "feat: generate_ai_graphic_post — orchestratore funzione pubblica"
```

---

### Task 7: Integrazione nel pipeline settimanale (`content_pipeline.py`)

**Files:**
- Modify: `src/social/content_pipeline.py` (funzione `run_weekly_pipeline`)

- [ ] **Step 1: Aggiungi il blocco ai_graphic alla fine di `run_weekly_pipeline`**

Aggiungi dopo il loop `for plan in selected:` (prima di `logger.info(f"Tenant {tenant_id}: piano settimana...")`):

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/social/content_pipeline.py
git commit -m "feat: integra ai_graphic nel pipeline settimanale — sabato, status draft"
```

---

### Task 8: Script di test con dati reali Amati

**Files:**
- Create: `test_ai_graphic.py` (root del progetto)

Questo script legge il profilo brand di Amati da Supabase, genera 3 post con categorie diverse, salva le immagini localmente. **Non tocca il DB di produzione.**

- [ ] **Step 1: Crea lo script**

```python
"""
Test integrazione ai_graphic con dati reali del tenant Amati.
Genera 3 post (tip_beauty, spotlight, ispirazione) e salva le immagini in test_output/ai_graphic/.
NON scrive nulla sul DB di produzione.

Uso:
    python test_ai_graphic.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.supabase_client import get_supabase
from src.social.gemini_social import (
    generate_ai_graphic_post,
    _get_next_ai_graphic_category,
    AI_GRAPHIC_CATEGORIES,
)

TENANT_API_KEY = "Amati_test_1"
OUTPUT_DIR = "test_output/ai_graphic"
CATEGORIES_TO_TEST = ["tip_beauty", "spotlight", "ispirazione"]


def test_rotation_logic():
    """Test puro della logica di rotazione — nessuna API call."""
    print("\n=== TEST ROTAZIONE (no API) ===")

    assert _get_next_ai_graphic_category({}) == "tip_beauty"
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": None}) == "tip_beauty"
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": "tip_beauty"}) == "spotlight"
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": "spotlight"}) == "stagionale"
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": "stagionale"}) == "ispirazione"
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": "ispirazione"}) == "tip_beauty"
    assert _get_next_ai_graphic_category({"last_ai_graphic_category": "xyz"}) == "tip_beauty"

    print("✓ Tutti i test rotazione passati")


async def generate_for_category(tenant: dict, category: str) -> None:
    """Genera un post per la categoria specificata (override rotazione per il test)."""
    print(f"\n--- Categoria: {category} ---")

    # Override: imposta last_ai_graphic_category al precedente per ottenere la categoria voluta
    tenant_copy = dict(tenant)
    sp = dict(tenant.get("social_profile") or {})
    idx = AI_GRAPHIC_CATEGORIES.index(category)
    prev = AI_GRAPHIC_CATEGORIES[(idx - 1) % len(AI_GRAPHIC_CATEGORIES)]
    sp["last_ai_graphic_category"] = prev
    tenant_copy["social_profile"] = sp

    result = await generate_ai_graphic_post(tenant_copy)

    print(f"Categoria usata : {result['ai_graphic_category']}")
    print(f"Concept         : {result.get('caption_text', '')[:80]}...")
    print(f"Caption         :\n{result['caption_text']}")
    print(f"Hashtag ({len(result['hashtags'])}): {' '.join(result['hashtags'][:6])}...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if result.get("feed_bytes"):
        path = f"{OUTPUT_DIR}/feed_{category}.jpg"
        with open(path, "wb") as f:
            f.write(result["feed_bytes"])
        size_kb = len(result["feed_bytes"]) // 1024
        print(f"✓ Feed salvato  : {path} ({size_kb} KB)")
    else:
        print(f"⚠  Nessuna immagine feed per {category}")

    if result.get("story_bytes"):
        path = f"{OUTPUT_DIR}/story_{category}.jpg"
        with open(path, "wb") as f:
            f.write(result["story_bytes"])
        size_kb = len(result["story_bytes"]) // 1024
        print(f"✓ Story salvata : {path} ({size_kb} KB)")


async def main():
    print("=" * 50)
    print("TEST AI GRAPHIC — Tenant Amati")
    print("=" * 50)

    # Test rotazione (puro, no API)
    test_rotation_logic()

    # Carica profilo Amati da Supabase
    sb = get_supabase()
    res = (
        sb.table("tenants")
        .select(
            "id, name, display_name, bio, logo_url, "
            "theme_primary_color, theme_secondary_color, social_profile"
        )
        .eq("content_api_key", TENANT_API_KEY)
        .maybe_single()
        .execute()
    )
    tenant = res.data
    if not tenant:
        print(f"\n✗ Tenant con api_key '{TENANT_API_KEY}' non trovato in Supabase")
        return

    name = tenant.get("display_name") or tenant.get("name")
    sp = tenant.get("social_profile") or {}
    print(f"\n✓ Tenant caricato: {name}")
    print(f"  Colori: {tenant.get('theme_primary_color')} / {tenant.get('theme_secondary_color')}")
    print(f"  Stile : {sp.get('photo_style')} — {sp.get('visual_style')}")
    print(f"  Tono  : {sp.get('tone_of_voice')}")

    # Genera 3 post con categorie diverse (le immagini NON vengono salvate su DB)
    for category in CATEGORIES_TO_TEST:
        await generate_for_category(tenant, category)

    print(f"\n{'=' * 50}")
    print(f"OUTPUT: {OUTPUT_DIR}/")
    print("Apri le immagini per valutare la qualità prima del deploy.")
    print("Se ok → Task 9 (push produzione). Se da migliorare → aggiusta i prompt.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Esegui il test**

```bash
cd "C:\Users\scate\OneDrive\Desktop\Gestionale_Estetiste\Social_Content_AI\backend"
python test_ai_graphic.py
```

Atteso:
- `✓ Tutti i test rotazione passati`
- Tenant Amati caricato con colori e stile
- 3 caption stampate nel terminale
- 6 file salvati in `test_output/ai_graphic/` (feed + story per ogni categoria)

- [ ] **Step 3: Apri le immagini e valuta**

Apri manualmente:
- `test_output/ai_graphic/feed_tip_beauty.jpg`
- `test_output/ai_graphic/feed_spotlight.jpg`
- `test_output/ai_graphic/feed_ispirazione.jpg`

Criteri di valutazione:
- Grafica di design (non foto realistica) ✓/✗
- Testo italiano leggibile ✓/✗
- Colori coerenti con brand Amati ✓/✗
- Stile adatto a un centro estetico ✓/✗

**Se la qualità non è soddisfacente:** aggiusta i prompt in `_generate_ai_graphic_image` (Task 5 Step 1) e riesegui il test. **Non procedere al Task 9 senza approvazione.**

---

### Task 9: Push in produzione (solo dopo approvazione qualità)

**Prerequisito:** immagini di test valutate e approvate.

- [ ] **Step 1: Commit finale del test script**

```bash
cd "C:\Users\scate\OneDrive\Desktop\Gestionale_Estetiste\Social_Content_AI\backend"
git add test_ai_graphic.py test_rotation.py
git commit -m "test: script integrazione ai_graphic con Amati"
```

- [ ] **Step 2: Push su master → Railway deploya**

```bash
git push https://<PAT>@github.com/sololabsrls-debug/Social_Content_AI.git master
# PAT disponibile in CLAUDE.md alla root del Bot_Estetiste
```

- [ ] **Step 3: Verifica deploy Railway**

Vai su Railway dashboard → verifica che il deploy sia verde.

- [ ] **Step 4: Test pipeline manuale**

Chiama l'endpoint di generazione manuale per verificare che tutto funzioni end-to-end in produzione:

```bash
curl -X POST https://web-production-d1724.up.railway.app/social/generate-week \
  -H "X-API-Key: Amati_test_1" \
  -H "Content-Type: application/json" \
  -d '{"week_start": "2026-04-07"}'
```

Atteso: risposta con `records_created` incrementato di 1 rispetto ai soli post da appuntamenti, e il nuovo record con `archetype=ai_graphic` visibile nel frontend con status `draft`.
