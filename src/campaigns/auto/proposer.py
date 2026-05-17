# src/campaigns/auto/proposer.py
import json
import logging
import os
from calendar import monthrange
from datetime import datetime, timezone

import anthropic

from src.supabase_client import get_supabase

logger = logging.getLogger("AUTO.proposer")

SEASON_BY_MONTH = {
    12: "inverno", 1: "inverno", 2: "inverno",
    3: "primavera", 4: "primavera", 5: "primavera",
    6: "estate", 7: "estate", 8: "estate",
    9: "autunno", 10: "autunno", 11: "autunno",
}

SEASON_HINTS = {
    "inverno": "riattivazione clienti, prodotti idratanti, trattamenti viso nutrienti",
    "primavera": "pulizie viso, epilazione, rinnovamento pelle",
    "estate": "epilazione gambe, pedicure, protezione solare, trattamenti corpo",
    "autunno": "trattamenti corpo, prodotti riparatori post-estate",
}

REQUIRED_FIELDS = {
    "service_product":  ["service_id", "product_id", "bundle_price", "original_price"],
    "service_only":     ["service_id", "bundle_price", "original_price"],
    "product_only":     ["product_id", "bundle_price", "original_price"],
    "multi_session":    ["service_id", "session_count", "bundle_price", "original_price"],
    "service_service":  ["service_id_1", "service_id_2", "bundle_price", "original_price"],
    "product_bundle":   ["product_ids", "bundle_price", "original_price"],
    "seasonal":         ["bundle_price", "original_price", "theme"],
    "reactivation":     ["service_id", "inactive_months_min", "discount_pct"],
}


def generate_proposals(tenant_id: str, plan_id: str, month: int, year: int, count: int = 8) -> int:
    """
    Genera count proposte AI per il piano mensile.
    Ritorna numero di proposte salvate con successo.
    """
    sb = get_supabase()

    # Fetch context
    services = _fetch_services(sb, tenant_id)
    products = _fetch_products(sb, tenant_id)
    existing_bundles = _fetch_existing_bundles(sb, tenant_id)
    inactive_count = _count_inactive_clients(sb, tenant_id)
    tenant_res = sb.table("tenants").select("name, display_name").eq("id", tenant_id).limit(1).execute()
    tenant = (tenant_res.data or [{}])[0]
    tenant_name = tenant.get("display_name") or tenant.get("name") or "il centro"

    season = SEASON_BY_MONTH.get(month, "primavera")
    season_hint = SEASON_HINTS[season]
    _, days_in_month = monthrange(year, month)

    month_names = ["", "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
                   "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"]
    month_name = month_names[month]

    services_str = "\n".join(
        f"- {s['name']} (id: {s['id']}, prezzo: €{s.get('price', 'N/D')})" for s in services[:20]
    )
    products_str = "\n".join(
        f"- {p['name']} (id: {p['id']}, prezzo: €{p.get('sale_price', p.get('price', 'N/D'))})"
        for p in products[:20]
    )
    bundles_str = ", ".join(b.get("name", "") for b in existing_bundles[:10]) or "nessuno"

    ai_request_count = count + 4  # buffer per compensare proposte che non passano validazione

    prompt = f"""Sei un esperto di marketing per centri estetici italiani.

Centro: {tenant_name}
Mese: {month_name} {year}
Stagione: {season} — trend: {season_hint}
Clienti inattivi (>3 mesi): {inactive_count}

Servizi disponibili:
{services_str if services_str else "nessun servizio caricato"}

Prodotti disponibili:
{products_str if products_str else "nessun prodotto caricato"}

Bundle già configurati (da non ripetere identici): {bundles_str}

Genera esattamente {ai_request_count} proposte di campagna promozionale per {month_name}.
Ogni proposta deve usare uno di questi tipi: service_product, service_only, product_only, multi_session, service_service, product_bundle, seasonal, reactivation.
Varia i tipi — non usare lo stesso tipo più di 2-3 volte.

IMPORTANTE — per ogni promo_type includi TUTTI i campi richiesti:
- service_product: service_id, product_id, bundle_price, original_price
- service_only: service_id, bundle_price, original_price
- product_only: product_id, bundle_price, original_price
- multi_session: service_id, session_count, bundle_price, original_price
- service_service: service_id_1, service_id_2, bundle_price, original_price
- product_bundle: product_ids (array), bundle_price, original_price
- seasonal: bundle_price, original_price, theme
- reactivation: service_id, inactive_months_min, discount_pct

Rispondi SOLO con un array JSON valido, senza testo extra:
[
  {{
    "promo_type": "service_product",
    "display_name": "Nome breve promo",
    "display_desc": "Descrizione 1-2 righe per l'estetista",
    "promo_config": {{
      "service_id": "<usa id reale dai servizi sopra>",
      "product_id": "<usa id reale dai prodotti sopra>",
      "bundle_price": 69.0,
      "original_price": 89.0
    }}
  }}
]

Usa SEMPRE id reali dai servizi/prodotti forniti."""

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Rimuovi eventuali backtick markdown
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        proposals_data = json.loads(raw)
    except Exception as exc:
        logger.error("AI proposal generation failed for tenant %s: %s", tenant_id, exc)
        return 0

    _, last_day = monthrange(year, month)
    expires_at = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc).isoformat()

    saved = 0
    for item in proposals_data:
        promo_type = item.get("promo_type", "")
        if promo_type not in REQUIRED_FIELDS:
            logger.warning("Unknown promo_type %s, skipping", promo_type)
            continue
        config = item.get("promo_config", {})
        missing = [f for f in REQUIRED_FIELDS[promo_type] if f not in config]
        if missing:
            logger.warning("Proposal missing fields %s for type %s, skipping", missing, promo_type)
            continue
        try:
            sb.table("auto_campaign_proposals").insert({
                "tenant_id": tenant_id,
                "plan_id": plan_id,
                "promo_type": promo_type,
                "promo_config": config,
                "display_name": item.get("display_name", promo_type),
                "display_desc": item.get("display_desc", ""),
                "status": "pending",
                "source": "ai",
                "expires_at": expires_at,
            }).execute()
            saved += 1
        except Exception as exc:
            logger.error("Failed to save proposal: %s", exc)

    sb.table("auto_campaign_plans").update({"proposals_count": saved}).eq("id", plan_id).execute()
    logger.info("Generated %d proposals for plan %s (tenant %s)", saved, plan_id, tenant_id)
    return saved


def _fetch_services(sb, tenant_id: str) -> list:
    res = sb.table("services").select("id, name, price") \
        .eq("tenant_id", tenant_id).eq("is_active", True).limit(30).execute()
    return res.data or []


def _fetch_products(sb, tenant_id: str) -> list:
    res = sb.table("products").select("id, name, sale_price") \
        .eq("tenant_id", tenant_id).eq("product_type", "retail").limit(30).execute()
    return res.data or []


def _fetch_existing_bundles(sb, tenant_id: str) -> list:
    res = sb.table("bundles").select("name") \
        .eq("tenant_id", tenant_id).eq("is_active", True).limit(20).execute()
    return res.data or []


def _count_inactive_clients(sb, tenant_id: str) -> int:
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    try:
        res = sb.table("clients").select("id", count="exact") \
            .eq("tenant_id", tenant_id) \
            .lt("last_visit_at", cutoff) \
            .execute()
        return res.count or 0
    except Exception:
        return 0
