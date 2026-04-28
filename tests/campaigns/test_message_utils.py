from src.campaigns.message_utils import (
    extract_first_name,
    normalize_campaign_message_template,
    render_campaign_message,
)


def test_extract_first_name_uses_last_token():
    assert extract_first_name("Scatena Lorenzo") == "Lorenzo"


def test_normalize_campaign_message_template_handles_literal_nome():
    normalized = normalize_campaign_message_template('Ciao "nome", ti aspettiamo in centro.')
    assert normalized == "Ciao {{nome}}, ti aspettiamo in centro."


def test_render_campaign_message_injects_real_first_name():
    rendered = render_campaign_message(
        "Gentile {{nome}}, abbiamo una promo per te.",
        "Rossi Marta",
    )
    assert rendered == "Gentile Marta, abbiamo una promo per te."
