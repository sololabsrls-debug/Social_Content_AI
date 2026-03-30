"""
Test integrazione ai_graphic con dati reali del tenant Amati.
Genera 3 post (tip_beauty, spotlight, ispirazione) e salva immagini in test_output/ai_graphic/.
NON scrive nulla sul DB di produzione.

Uso:
    python test_ai_graphic.py
"""
import asyncio
import os
import sys

# Fix encoding per terminale Windows (cp1252 non supporta emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

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

    print("OK Tutti i test rotazione passati")


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
    print(f"Caption         :\n{result['caption_text']}")
    print(f"Hashtag ({len(result['hashtags'])}): {' '.join(result['hashtags'][:6])}...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if result.get("feed_bytes"):
        path = f"{OUTPUT_DIR}/feed_{category}.jpg"
        with open(path, "wb") as f:
            f.write(result["feed_bytes"])
        size_kb = len(result["feed_bytes"]) // 1024
        print(f"OK Feed salvato  : {path} ({size_kb} KB)")
    else:
        print(f"WARN Nessuna immagine feed per {category}")

    if result.get("story_bytes"):
        path = f"{OUTPUT_DIR}/story_{category}.jpg"
        with open(path, "wb") as f:
            f.write(result["story_bytes"])
        size_kb = len(result["story_bytes"]) // 1024
        print(f"OK Story salvata : {path} ({size_kb} KB)")


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
        print(f"\nERRORE: Tenant con api_key '{TENANT_API_KEY}' non trovato in Supabase")
        return

    name = tenant.get("display_name") or tenant.get("name")
    sp = tenant.get("social_profile") or {}
    print(f"\nOK Tenant caricato: {name}")
    print(f"  Colori: {tenant.get('theme_primary_color')} / {tenant.get('theme_secondary_color')}")
    print(f"  Stile : {sp.get('photo_style')} — {sp.get('visual_style')}")
    print(f"  Tono  : {sp.get('tone_of_voice')}")

    # Genera 3 post con categorie diverse (le immagini NON vengono salvate su DB)
    for category in CATEGORIES_TO_TEST:
        await generate_for_category(tenant, category)

    print(f"\n{'=' * 50}")
    print(f"OUTPUT: {OUTPUT_DIR}/")
    print("Apri le immagini per valutare la qualita prima del deploy.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
