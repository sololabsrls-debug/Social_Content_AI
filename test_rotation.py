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

    print("[OK] Tutti i test rotazione passati")

if __name__ == "__main__":
    test_rotation()
