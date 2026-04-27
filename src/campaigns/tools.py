"""
22 parameterized Supabase query tools for the Campaign AI agent.
All functions take tenant_id as first argument and use service-role client (bypasses RLS).
Results are capped to avoid overwhelming Claude's context.
"""

import logging
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Optional

from src.supabase_client import get_supabase

logger = logging.getLogger("CAMPAIGNS.tools")

# ── Clienti ────────────────────────────────────────────────────────


def get_clients_overview(tenant_id: str) -> dict[str, int]:
    """Totale clienti, quanti hanno WhatsApp, quanti hanno dato consenso WA."""
    sb = get_supabase()
    res = (
        sb.table("clients")
        .select("id, whatsapp_phone, consent_wa")
        .eq("tenant_id", tenant_id)
        .execute()
    )
    rows = res.data or []
    return {
        "total": len(rows),
        "with_wa": sum(1 for r in rows if r.get("whatsapp_phone")),
        "with_consent": sum(1 for r in rows if r.get("consent_wa") and r.get("whatsapp_phone")),
    }


def get_clients_reachable_wa(tenant_id: str) -> list[dict]:
    """Clienti con whatsapp_phone e consent_wa = true."""
    sb = get_supabase()
    res = (
        sb.table("clients")
        .select("id, name, whatsapp_phone, last_appointment_at, ltv, dob")
        .eq("tenant_id", tenant_id)
        .eq("consent_wa", True)
        .not_.is_("whatsapp_phone", "null")
        .limit(200)
        .execute()
    )
    return res.data or []


def get_clients_by_service(
    tenant_id: str,
    service_name: str,
    min_visits: int = 1,
    months_back: int = 6,
) -> list[dict]:
    """Clienti che hanno fatto un servizio almeno N volte negli ultimi M mesi."""
    sb = get_supabase()
    cutoff = (date.today() - timedelta(days=30 * months_back)).isoformat()
    appts = (
        sb.table("appointments")
        .select("client_id, service:services(name)")
        .eq("tenant_id", tenant_id)
        .gte("start_at", cutoff)
        .in_("status", ["confirmed", "completed"])
        .execute()
    )
    rows = appts.data or []
    service_lower = service_name.lower()
    counts: Counter = Counter()
    for r in rows:
        svc = r.get("service") or {}
        svc_name = (svc.get("name") or "").lower()
        if service_lower in svc_name and r.get("client_id"):
            counts[r["client_id"]] += 1
    qualifying = [cid for cid, cnt in counts.items() if cnt >= min_visits]
    if not qualifying:
        return []
    clients = (
        sb.table("clients")
        .select("id, name, whatsapp_phone, consent_wa, last_appointment_at, ltv")
        .eq("tenant_id", tenant_id)
        .in_("id", qualifying[:100])
        .execute()
    )
    return clients.data or []


def get_inactive_clients(
    tenant_id: str,
    days_inactive: int = 90,
    exclude_with_future: bool = True,
) -> list[dict]:
    """Clienti senza appuntamenti negli ultimi N giorni."""
    sb = get_supabase()
    cutoff = (date.today() - timedelta(days=days_inactive)).isoformat()
    res = (
        sb.table("clients")
        .select("id, name, whatsapp_phone, consent_wa, last_appointment_at, ltv")
        .eq("tenant_id", tenant_id)
        .eq("consent_wa", True)
        .not_.is_("whatsapp_phone", "null")
        .lt("last_appointment_at", cutoff)
        .limit(150)
        .execute()
    )
    clients = res.data or []
    if not exclude_with_future or not clients:
        return clients
    future_ids = set(get_clients_with_future_appointments(tenant_id))
    return [c for c in clients if c["id"] not in future_ids]


def get_clients_with_future_appointments(tenant_id: str) -> list[str]:
    """Client IDs che hanno almeno un appuntamento futuro."""
    sb = get_supabase()
    today = date.today().isoformat()
    res = (
        sb.table("appointments")
        .select("client_id")
        .eq("tenant_id", tenant_id)
        .gte("start_at", today)
        .in_("status", ["confirmed", "pending"])
        .execute()
    )
    return list({r["client_id"] for r in (res.data or []) if r.get("client_id")})


def get_clients_by_ltv(
    tenant_id: str,
    min_ltv: float,
    max_ltv: Optional[float] = None,
) -> list[dict]:
    """Clienti per fascia di valore (LTV)."""
    sb = get_supabase()
    q = (
        sb.table("clients")
        .select("id, name, whatsapp_phone, consent_wa, ltv, last_appointment_at")
        .eq("tenant_id", tenant_id)
        .eq("consent_wa", True)
        .not_.is_("whatsapp_phone", "null")
        .gte("ltv", min_ltv)
    )
    if max_ltv is not None:
        q = q.lte("ltv", max_ltv)
    res = q.limit(100).execute()
    return res.data or []


def get_clients_with_birthday(tenant_id: str, days_ahead: int = 14) -> list[dict]:
    """Clienti con compleanno nei prossimi N giorni (ignora anno)."""
    sb = get_supabase()
    res = (
        sb.table("clients")
        .select("id, name, whatsapp_phone, consent_wa, dob")
        .eq("tenant_id", tenant_id)
        .eq("consent_wa", True)
        .not_.is_("whatsapp_phone", "null")
        .not_.is_("dob", "null")
        .limit(200)
        .execute()
    )
    today = date.today()
    result = []
    for c in (res.data or []):
        dob_str = c.get("dob")
        if not dob_str:
            continue
        try:
            dob = date.fromisoformat(dob_str[:10])
            birthday_this_year = dob.replace(year=today.year)
            if birthday_this_year < today:
                birthday_this_year = dob.replace(year=today.year + 1)
            if 0 <= (birthday_this_year - today).days <= days_ahead:
                result.append(c)
        except ValueError:
            continue
    return result


def get_recently_contacted_clients(tenant_id: str, days_back: int = 7) -> list[str]:
    """Client IDs contattati via WA negli ultimi N giorni (per cooldown)."""
    sb = get_supabase()
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    res = (
        sb.table("clients")
        .select("id")
        .eq("tenant_id", tenant_id)
        .gte("ultima_interazione_wa", cutoff)
        .execute()
    )
    return [r["id"] for r in (res.data or [])]


def get_clients_by_tag(tenant_id: str, tag_name: str) -> list[dict]:
    """Clienti con un tag specifico."""
    sb = get_supabase()
    tag_res = (
        sb.table("client_tags")
        .select("id")
        .eq("tenant_id", tenant_id)
        .ilike("name", f"%{tag_name}%")
        .limit(1)
        .execute()
    )
    tags = tag_res.data or []
    if not tags:
        return []
    tag_id = tags[0]["id"]
    links = (
        sb.table("client_tag_links")
        .select("client_id")
        .eq("tag_id", tag_id)
        # tag_id already scoped to tenant; client_tag_links has no tenant_id column
        .execute()
    )
    client_ids = [r["client_id"] for r in (links.data or [])]
    if not client_ids:
        return []
    clients = (
        sb.table("clients")
        .select("id, name, whatsapp_phone, consent_wa, last_appointment_at, ltv")
        .eq("tenant_id", tenant_id)
        .in_("id", client_ids[:100])
        .execute()
    )
    return clients.data or []


def get_clients_never_returned(tenant_id: str, after_first_visit_days: int = 60) -> list[dict]:
    """Clienti con una sola visita e non tornate entro N giorni."""
    sb = get_supabase()
    cutoff = (date.today() - timedelta(days=after_first_visit_days)).isoformat()
    res = (
        sb.table("appointments")
        .select("client_id")
        .eq("tenant_id", tenant_id)
        .in_("status", ["confirmed", "completed"])
        .execute()
    )
    counts: Counter = Counter(
        r["client_id"] for r in (res.data or []) if r.get("client_id")
    )
    single_visit_ids = [cid for cid, cnt in counts.items() if cnt == 1]
    if not single_visit_ids:
        return []
    clients = (
        sb.table("clients")
        .select("id, name, whatsapp_phone, consent_wa, last_appointment_at")
        .eq("tenant_id", tenant_id)
        .eq("consent_wa", True)
        .not_.is_("whatsapp_phone", "null")
        .in_("id", single_visit_ids[:100])
        .lt("last_appointment_at", cutoff)
        .execute()
    )
    return clients.data or []


# ── Appuntamenti ───────────────────────────────────────────────────


def get_appointments_gaps(
    tenant_id: str,
    date_str: str,
    staff_id: Optional[str] = None,
) -> list[dict]:
    """Slot prenotati in una data specifica (l'agente inferisce i buchi)."""
    sb = get_supabase()
    q = (
        sb.table("appointments")
        .select("start_at, end_at, staff_id")
        .eq("tenant_id", tenant_id)
        .gte("start_at", f"{date_str}T00:00:00")
        .lte("start_at", f"{date_str}T23:59:59")
        .in_("status", ["confirmed", "pending"])
    )
    if staff_id:
        q = q.eq("staff_id", staff_id)
    res = q.order("start_at").execute()
    booked = res.data or []
    return [{"start": r["start_at"], "end": r["end_at"]} for r in booked]


def get_appointments_by_service(
    tenant_id: str,
    service_name: str,
    months_back: int = 6,
) -> dict:
    """Volume appuntamenti per servizio negli ultimi N mesi."""
    sb = get_supabase()
    cutoff = (date.today() - timedelta(days=30 * months_back)).isoformat()
    res = (
        sb.table("appointments")
        .select("id, start_at, service:services(name)")
        .eq("tenant_id", tenant_id)
        .gte("start_at", cutoff)
        .in_("status", ["confirmed", "completed"])
        .limit(2000)
        .execute()
    )
    service_lower = service_name.lower()
    matching = [
        r for r in (res.data or [])
        if service_lower in ((r.get("service") or {}).get("name") or "").lower()
    ]
    return {"service_name": service_name, "count": len(matching), "months_back": months_back}


def get_busiest_services(tenant_id: str, months_back: int = 3, limit: int = 5) -> list[dict]:
    """Servizi più prenotati negli ultimi N mesi."""
    sb = get_supabase()
    cutoff = (date.today() - timedelta(days=30 * months_back)).isoformat()
    res = (
        sb.table("appointments")
        .select("service:services(name)")
        .eq("tenant_id", tenant_id)
        .gte("start_at", cutoff)
        .in_("status", ["confirmed", "completed"])
        .execute()
    )
    counts: Counter = Counter(
        (r.get("service") or {}).get("name")
        for r in (res.data or [])
        if (r.get("service") or {}).get("name")
    )
    return [{"service": name, "count": cnt} for name, cnt in counts.most_common(limit)]


def get_upcoming_week_availability(tenant_id: str) -> dict:
    """Slot prenotati nella settimana corrente."""
    sb = get_supabase()
    today = date.today()
    week_end = today + timedelta(days=7)
    res = (
        sb.table("appointments")
        .select("start_at, end_at, staff_id")
        .eq("tenant_id", tenant_id)
        .gte("start_at", today.isoformat())
        .lte("start_at", week_end.isoformat())
        .in_("status", ["confirmed", "pending"])
        .order("start_at")
        .execute()
    )
    slots = res.data or []
    return {
        "period_start": today.isoformat(),
        "period_end": week_end.isoformat(),
        "booked_slots": [{"start": r["start_at"], "end": r["end_at"]} for r in slots],
        "booked_count": len(slots),
    }


# ── Servizi ────────────────────────────────────────────────────────


def get_services_list(tenant_id: str) -> list[dict]:
    """Lista tutti i servizi del centro."""
    sb = get_supabase()
    res = (
        sb.table("services")
        .select("id, name, descrizione_breve, duration_min")
        .eq("tenant_id", tenant_id)
        .execute()
    )
    return res.data or []


def get_service_details(tenant_id: str, service_name: str) -> Optional[dict]:
    """Dettaglio completo di un servizio."""
    sb = get_supabase()
    res = (
        sb.table("services")
        .select("id, name, descrizione_breve, descrizione_completa, benefici, prodotti_utilizzati, duration_min")
        .eq("tenant_id", tenant_id)
        .ilike("name", f"%{service_name}%")
        .limit(1)
        .execute()
    )
    data = res.data or []
    return data[0] if data else None


# ── Analytics ──────────────────────────────────────────────────────


def get_client_retention_rate(tenant_id: str, months_back: int = 6) -> dict:
    """Percentuale clienti tornate almeno 2 volte nel periodo."""
    sb = get_supabase()
    cutoff = (date.today() - timedelta(days=30 * months_back)).isoformat()
    res = (
        sb.table("appointments")
        .select("client_id")
        .eq("tenant_id", tenant_id)
        .gte("start_at", cutoff)
        .in_("status", ["confirmed", "completed"])
        .execute()
    )
    counts: Counter = Counter(
        r["client_id"] for r in (res.data or []) if r.get("client_id")
    )
    total = len(counts)
    returned = sum(1 for cnt in counts.values() if cnt >= 2)
    rate = round(returned / total * 100, 1) if total > 0 else 0.0
    return {"total_clients": total, "returned": returned, "retention_rate_pct": rate}


def get_avg_days_between_visits(
    tenant_id: str,
    service_name: Optional[str] = None,
) -> dict:
    """Frequenza media di ritorno (in giorni)."""
    sb = get_supabase()
    cutoff = (date.today() - timedelta(days=365)).isoformat()
    q = (
        sb.table("appointments")
        .select("client_id, start_at, service:services(name)")
        .eq("tenant_id", tenant_id)
        .gte("start_at", cutoff)
        .in_("status", ["confirmed", "completed"])
        .order("client_id")
        .order("start_at")
    )
    res = q.execute()
    rows = res.data or []
    if service_name:
        svc_lower = service_name.lower()
        rows = [r for r in rows if svc_lower in ((r.get("service") or {}).get("name") or "").lower()]
    by_client: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r.get("client_id") and r.get("start_at"):
            by_client[r["client_id"]].append(r["start_at"])
    gaps = []
    for dates in by_client.values():
        if len(dates) < 2:
            continue
        sorted_dates = sorted(dates)
        for i in range(1, len(sorted_dates)):
            d1 = datetime.fromisoformat(sorted_dates[i - 1][:10])
            d2 = datetime.fromisoformat(sorted_dates[i][:10])
            gaps.append((d2 - d1).days)
    avg = round(sum(gaps) / len(gaps), 1) if gaps else 0.0
    return {"avg_days_between_visits": avg, "sample_size": len(gaps), "service_filter": service_name}


def get_whatsapp_campaign_history(tenant_id: str, days_back: int = 30) -> list[dict]:
    """Campagne WA inviate negli ultimi N giorni."""
    sb = get_supabase()
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    res = (
        sb.table("wa_campaigns")
        .select("id, status, objective, recipients_count, sent_at, created_at")
        .eq("tenant_id", tenant_id)
        .gte("created_at", cutoff)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    return res.data or []


# ── Centro ─────────────────────────────────────────────────────────


def get_tenant_profile(tenant_id: str) -> dict:
    """Profilo del centro."""
    sb = get_supabase()
    res = (
        sb.table("tenants")
        .select("id, name, display_name, bio, logo_url, social_profile")
        .eq("id", tenant_id)
        .limit(1)
        .execute()
    )
    data = res.data or []
    return data[0] if data else {}


def get_opening_hours(tenant_id: str) -> list[dict]:
    """Orari di apertura del centro."""
    sb = get_supabase()
    res = (
        sb.table("opening_hours")
        .select("day_of_week, open_time, close_time, is_closed")
        .eq("tenant_id", tenant_id)
        .order("day_of_week")
        .execute()
    )
    return res.data or []


def get_staff_list(tenant_id: str) -> list[dict]:
    """Lista staff del centro."""
    sb = get_supabase()
    res = (
        sb.table("staff")
        .select("id, name, role")
        .eq("tenant_id", tenant_id)
        .execute()
    )
    return res.data or []


# ── Tool schema definitions for Claude ────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_clients_overview",
        "description": "Conta il totale clienti del centro, quanti hanno WhatsApp e quanti hanno dato consenso per messaggi promozionali.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_clients_reachable_wa",
        "description": "Restituisce la lista completa di clienti raggiungibili via WhatsApp (numero presente + consenso dato).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_clients_by_service",
        "description": "Trova clienti che hanno effettuato un servizio specifico almeno N volte negli ultimi M mesi.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service_name": {"type": "string", "description": "Nome del servizio (es. 'manicure', 'ceretta')"},
                "min_visits": {"type": "integer", "description": "Numero minimo di visite richieste", "default": 1},
                "months_back": {"type": "integer", "description": "Quanti mesi indietro cercare", "default": 6},
            },
            "required": ["service_name"],
        },
    },
    {
        "name": "get_inactive_clients",
        "description": "Trova clienti che non hanno appuntamenti da almeno N giorni. Esclude automaticamente chi ha già appuntamenti futuri.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_inactive": {"type": "integer", "description": "Giorni di inattività minimi", "default": 90},
                "exclude_with_future": {"type": "boolean", "description": "Escludere chi ha già un appuntamento futuro", "default": True},
            },
            "required": [],
        },
    },
    {
        "name": "get_clients_with_future_appointments",
        "description": "Restituisce gli ID delle clienti che hanno già un appuntamento futuro confermato o in attesa.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_clients_by_ltv",
        "description": "Trova clienti per fascia di valore (LTV — valore totale speso nel centro).",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_ltv": {"type": "number", "description": "LTV minimo in euro"},
                "max_ltv": {"type": "number", "description": "LTV massimo in euro (opzionale)"},
            },
            "required": ["min_ltv"],
        },
    },
    {
        "name": "get_clients_with_birthday",
        "description": "Trova clienti con compleanno nei prossimi N giorni.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_ahead": {"type": "integer", "description": "Giorni in avanti da oggi", "default": 14},
            },
            "required": [],
        },
    },
    {
        "name": "get_recently_contacted_clients",
        "description": "Restituisce gli ID delle clienti contattate via WhatsApp negli ultimi N giorni (per evitare spam).",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {"type": "integer", "description": "Quanti giorni indietro controllare", "default": 7},
            },
            "required": [],
        },
    },
    {
        "name": "get_clients_by_tag",
        "description": "Trova clienti con un tag specifico (es. 'vip', 'fidelizzata').",
        "input_schema": {
            "type": "object",
            "properties": {
                "tag_name": {"type": "string", "description": "Nome del tag"},
            },
            "required": ["tag_name"],
        },
    },
    {
        "name": "get_clients_never_returned",
        "description": "Trova clienti con una sola visita che non sono tornate entro N giorni dalla prima visita.",
        "input_schema": {
            "type": "object",
            "properties": {
                "after_first_visit_days": {"type": "integer", "description": "Giorni attesi prima del ritorno", "default": 60},
            },
            "required": [],
        },
    },
    {
        "name": "get_appointments_gaps",
        "description": "Mostra gli slot prenotati in una data specifica così da individuare i buchi liberi.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_str": {"type": "string", "description": "Data in formato YYYY-MM-DD"},
                "staff_id": {"type": "string", "description": "ID staff (opzionale)"},
            },
            "required": ["date_str"],
        },
    },
    {
        "name": "get_appointments_by_service",
        "description": "Conta quante volte è stato prenotato un servizio negli ultimi N mesi.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service_name": {"type": "string"},
                "months_back": {"type": "integer", "default": 6},
            },
            "required": ["service_name"],
        },
    },
    {
        "name": "get_busiest_services",
        "description": "Restituisce i servizi più prenotati negli ultimi N mesi.",
        "input_schema": {
            "type": "object",
            "properties": {
                "months_back": {"type": "integer", "default": 3},
                "limit": {"type": "integer", "default": 5},
            },
            "required": [],
        },
    },
    {
        "name": "get_upcoming_week_availability",
        "description": "Mostra gli slot già prenotati nella settimana corrente. Utile per campagne 'riempi i buchi'.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_services_list",
        "description": "Lista tutti i servizi offerti dal centro con nome e durata.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_service_details",
        "description": "Dettaglio completo di un servizio: descrizione, benefici, prodotti utilizzati.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service_name": {"type": "string"},
            },
            "required": ["service_name"],
        },
    },
    {
        "name": "get_client_retention_rate",
        "description": "Calcola la percentuale di clienti tornate almeno due volte nel periodo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "months_back": {"type": "integer", "default": 6},
            },
            "required": [],
        },
    },
    {
        "name": "get_avg_days_between_visits",
        "description": "Calcola la frequenza media di ritorno delle clienti in giorni, opzionalmente filtrata per servizio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service_name": {"type": "string", "description": "Filtra per servizio specifico (opzionale)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_whatsapp_campaign_history",
        "description": "Mostra le campagne WhatsApp inviate negli ultimi N giorni.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {"type": "integer", "default": 30},
            },
            "required": [],
        },
    },
    {
        "name": "get_tenant_profile",
        "description": "Profilo del centro estetico: nome, bio, stile, posizionamento.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_opening_hours",
        "description": "Orari di apertura del centro per ogni giorno della settimana.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_staff_list",
        "description": "Lista del personale del centro.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "propose_campaign",
        "description": (
            "Chiama OBBLIGATORIAMENTE questo tool quando hai finito l'analisi e sei pronto "
            "a proporre la campagna. Passa il motivo del target e il messaggio WhatsApp già "
            "scritto e pronto per l'invio."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_reason": {
                    "type": "string",
                    "description": (
                        "Spiegazione in 2-3 frasi del perché questo target è stato scelto "
                        "e perché la campagna avrà successo. Senza asterischi o trattini."
                    ),
                },
                "wa_message": {
                    "type": "string",
                    "description": (
                        "Il messaggio WhatsApp completo, pronto per l'invio. "
                        "Usa {{nome}} come segnaposto per il nome cliente. "
                        "Niente asterischi, niente trattini."
                    ),
                },
            },
            "required": ["target_reason", "wa_message"],
        },
    },
]

# Map tool name → function for execute_tool in agent.py
TOOL_FUNCTIONS: dict[str, Any] = {
    "get_clients_overview": get_clients_overview,
    "get_clients_reachable_wa": get_clients_reachable_wa,
    "get_clients_by_service": get_clients_by_service,
    "get_inactive_clients": get_inactive_clients,
    "get_clients_with_future_appointments": get_clients_with_future_appointments,
    "get_clients_by_ltv": get_clients_by_ltv,
    "get_clients_with_birthday": get_clients_with_birthday,
    "get_recently_contacted_clients": get_recently_contacted_clients,
    "get_clients_by_tag": get_clients_by_tag,
    "get_clients_never_returned": get_clients_never_returned,
    "get_appointments_gaps": get_appointments_gaps,
    "get_appointments_by_service": get_appointments_by_service,
    "get_busiest_services": get_busiest_services,
    "get_upcoming_week_availability": get_upcoming_week_availability,
    "get_services_list": get_services_list,
    "get_service_details": get_service_details,
    "get_client_retention_rate": get_client_retention_rate,
    "get_avg_days_between_visits": get_avg_days_between_visits,
    "get_whatsapp_campaign_history": get_whatsapp_campaign_history,
    "get_tenant_profile": get_tenant_profile,
    "get_opening_hours": get_opening_hours,
    "get_staff_list": get_staff_list,
    "propose_campaign": lambda tenant_id, **_: {"status": "ok"},
}
