"""
Claude-powered campaign agent.
Runs a tool_use loop and yields SSE-compatible (event_type, data) tuples.
Saves campaign state to Supabase after each run.
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Optional

import anthropic

from src.campaigns.message_utils import normalize_campaign_message_template
from src.campaigns.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
from src.supabase_client import get_supabase

logger = logging.getLogger("CAMPAIGNS.agent")

SYSTEM_PROMPT = """Sei un assistente marketing esperto per centri estetici. Aiuti l'estetista a creare campagne WhatsApp efficaci, chiare e subito utilizzabili.

GUARDRAIL, Ambito di lavoro:
Puoi fare solo queste cose, analizzare dati del centro, proporre campagne WhatsApp, modificare target e messaggi, rispondere a domande sulla campagna in corso.
Se l'estetista chiede qualcosa fuori da questo ambito, prenotazioni, contabilita, social media, ricette, domande generali, rispondi con una sola frase, "Sono qui solo per le campagne WhatsApp. Dimmi come vuoi impostare la campagna e penso a tutto io."
Non spiegare, non scusarti, non divagare. Reindirizza subito.

Flusso di lavoro obbligatorio:
1. Usa i tool di analisi per capire i dati reali del centro, clienti, appuntamenti, servizi.
2. Quando hai finito l'analisi, chiama sempre il tool propose_campaign.
3. In propose_campaign compila sempre tutti questi campi, objective_summary, target_reason, wa_message, wa_message_variant, treatment_label.
4. Dopo propose_campaign, spiega in chat in modo chiaro, prima chi colpisci, poi perche, poi che messaggio invierai.

Gestione follow-up e modifiche:
Se devi aggiungere clienti, usa get_client_by_name per ogni cliente da aggiungere, poi chiama propose_campaign con target_client_names uguale alla lista completa, vecchi piu nuovi.
Se devi rimuovere clienti, usa get_client_by_name solo per i clienti da mantenere, poi chiama propose_campaign con target_client_names uguale alla lista finale.
Se devi cambiare target completamente, usa i tool di analisi normali, poi chiama propose_campaign.
Se devi cambiare solo il messaggio, chiama propose_campaign direttamente con il testo riscritto, senza rifare analisi clienti.
Ogni modifica deve sempre concludersi con propose_campaign.

Regola follow-up assoluta:
Se l'estetista chiede di aggiungere o rimuovere clienti specifici per nome, usa solo get_client_by_name.
Non chiamare mai get_clients_by_service, get_inactive_clients, get_clients_reachable_wa, get_clients_by_ltv, get_clients_with_birthday, get_clients_by_tag, get_clients_never_returned o altri tool di analisi generici.
Violazione uguale errore grave.

Regola assoluta sui nomi:
I nomi nel database sono in formato COGNOME NOME, esempio Scatena Lorenzo.
Nel messaggio WhatsApp usa sempre e obbligatoriamente {{nome}} come unico segnaposto.
Mai usare nome reale o cognome reale nel testo finale.

Regola chiarezza assoluta:
Se l'estetista chiede spiegazioni, rispondi in modo didascalico e specifico.
Spiega sempre quale dato hai usato, quale target hai scelto, perche lo hai scelto, e quale messaggio consigli.
Non usare frasi vaghe come "e un buon target" o "potrebbe funzionare".

Regola proposta diretta:
Non chiedere mai conferma o approvazione prima di chiamare propose_campaign.
Proponi sempre la campagna direttamente, qualunque sia il numero di clienti nel target.
Dopo propose_campaign puoi chiedere se vuole modifiche, mai prima.

Regole importanti:
Parla sempre in italiano.
Sii diretto, concreto e preciso.
Risposte brevi ma complete.
Non suggerire mai l'invio senza approvazione esplicita.
Fai poche domande, solo se strettamente necessarie.
Usa un tono caldo e professionale.
Non usare mai asterischi, trattini, grassetto, corsivo o altri simboli markdown. Usa frasi normali e virgole.
Il messaggio WhatsApp in propose_campaign deve contenere {{nome}} come unico segnaposto per il nome.
wa_message_variant deve essere una variante piu corta o piu premium del messaggio principale.
treatment_label deve essere solo il nome breve del trattamento o focus campagna, massimo 4 parole.
objective_summary deve essere una sintesi chiara dell'obiettivo attuale della campagna.
"""


def _dedupe_client_data(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    normalized: list[dict] = []
    for row in rows:
        phone = (row.get("phone") or row.get("whatsapp_phone") or "").strip()
        if not phone or phone in seen:
            continue
        seen.add(phone)
        normalized.append(
            {
                "phone": phone,
                "name": (row.get("name") or "").strip(),
            }
        )
    return normalized


def _get_named_client_edit_mode(last_user_text: str) -> str:
    text = (last_user_text or "").lower()
    replace_signals = (
        "rimuovi",
        "togli",
        "cancella",
        "escludi",
        "lascia solo",
        "mantieni solo",
        "tieni solo",
        "voglio solo",
    )
    append_signals = (
        "aggiungi",
        "includi",
        "inserisci",
        "unisci",
    )
    if any(signal in text for signal in replace_signals):
        return "replace"
    if any(signal in text for signal in append_signals):
        return "append"
    return "unknown"


def derive_canvas_update(tool_name: str, result: Any) -> Optional[dict]:
    """Derives a canvas block update from a tool result."""
    target_tools = {
        "get_clients_by_service",
        "get_inactive_clients",
        "get_clients_by_ltv",
        "get_clients_with_birthday",
        "get_clients_by_tag",
        "get_clients_never_returned",
        "get_client_by_name",
    }

    if tool_name == "get_clients_reachable_wa" and isinstance(result, list):
        examples = [
            {"id": c.get("id"), "name": c.get("name")}
            for c in result[:5]
            if c.get("name")
        ]
        return {
            "block": "target",
            "state": "analyzing",
            "data": {
                "reachable": len(result),
                "examples": examples,
            },
        }

    if tool_name == "get_recently_contacted_clients" and isinstance(result, list):
        return {
            "block": "target",
            "state": "analyzing",
            "data": {"excluded": len(result)},
        }

    if tool_name in target_tools and isinstance(result, list):
        examples = [
            {"id": c.get("id"), "name": c.get("name")}
            for c in result[:5]
            if c.get("name")
        ]
        return {
            "block": "target",
            "state": "analyzing",
            "data": {
                "count": len(result),
                "reachable": sum(1 for c in result if c.get("whatsapp_phone")),
                "examples": examples,
            },
        }

    if tool_name == "get_upcoming_week_availability" and isinstance(result, dict):
        return {
            "block": "target",
            "state": "analyzing",
            "data": {"booked_count": result.get("booked_count", 0)},
        }

    return None


async def execute_tool(tool_name: str, tool_input: dict, tenant_id: str) -> Any:
    """Calls the appropriate tool function with tenant_id injected."""
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        logger.warning("Unknown tool: %s", tool_name)
        return {"error": f"Tool {tool_name} not found"}
    try:
        return await asyncio.to_thread(fn, tenant_id=tenant_id, **tool_input)
    except Exception as exc:
        logger.error("Tool %s failed: %s", tool_name, exc)
        return {"error": str(exc)}


async def run_campaign_agent(
    messages: list[dict],
    tenant_id: str,
    campaign_id: Optional[str],
    initial_target_count: int = 0,
    existing_client_data: Optional[list[dict]] = None,
) -> AsyncGenerator[tuple[str, dict], None]:
    """
    Runs the Claude tool_use loop and yields (event_type, data) tuples.
    Event types: thinking | tool_call | canvas_update | message | done
    """
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    claude_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    if existing_client_data:
        names = [c.get("name", "") for c in existing_client_data if c.get("name")]
        if names and claude_messages and claude_messages[-1]["role"] == "user":
            context_note = (
                f"[SISTEMA, clienti ATTUALMENTE nel target: {', '.join(names)}. "
                "AGGIUNGERE, usa get_client_by_name per i nuovi, poi propose_campaign "
                "con target_client_names uguale a tutti, vecchi piu nuovi. "
                "RIMUOVERE, usa get_client_by_name solo per chi rimane, poi "
                "propose_campaign con target_client_names uguale a solo chi rimane.]"
            )
            last = claude_messages[-1]
            claude_messages[-1] = {
                "role": "user",
                "content": f"{context_note}\n\n{last['content']}",
            }

    canvas_state: dict[str, Any] = {}
    final_target_count = 0
    last_user_text = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )
    named_client_edit_mode = _get_named_client_edit_mode(last_user_text)
    existing_target_client_data = _dedupe_client_data(existing_client_data or [])
    target_client_data: list[dict] = []
    target_selection_touched = False
    named_lookup_seeded = False

    target_replace_tools = {
        "get_clients_by_service",
        "get_inactive_clients",
        "get_clients_by_ltv",
        "get_clients_with_birthday",
        "get_clients_by_tag",
        "get_clients_never_returned",
    }

    first_user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
    if first_user_text:
        canvas_state["objective"] = {"state": "analyzing", "data": {"text": first_user_text}}
        yield "canvas_update", {
            "block": "objective",
            "state": "analyzing",
            "data": {"text": first_user_text},
        }

    for _ in range(12):
        yield "thinking", {"text": "Elaboro la risposta..."}

        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=claude_messages,
        )

        text_parts: list[str] = []
        tool_calls: list[Any] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                if tc.name == "propose_campaign":
                    objective_summary = (tc.input.get("objective_summary") or "").strip()
                    reason_text = (tc.input.get("target_reason") or "").strip()
                    wa_message = (tc.input.get("wa_message") or "").strip()
                    wa_message_variant = (tc.input.get("wa_message_variant") or "").strip()
                    treatment_label = (tc.input.get("treatment_label") or "").strip()
                    target_client_names: list[str] = tc.input.get("target_client_names") or []

                    if objective_summary:
                        objective_data = {
                            **canvas_state.get("objective", {}).get("data", {}),
                            "text": objective_summary,
                        }
                        canvas_state["objective"] = {"state": "ready", "data": objective_data}
                        yield "canvas_update", {
                            "block": "objective",
                            "state": "ready",
                            "data": objective_data,
                        }

                    if target_client_names:
                        target_selection_touched = True
                        rebuilt_clients: list[dict] = []
                        for name in target_client_names:
                            found = await execute_tool("get_client_by_name", {"name": name}, tenant_id)
                            if isinstance(found, list):
                                rebuilt_clients.extend(found)
                        target_client_data = _dedupe_client_data(rebuilt_clients)

                        existing_target = canvas_state.get("target", {}).get("data", {})
                        target_update_data = {
                            **existing_target,
                            "count": len(target_client_data),
                            "reachable": len(target_client_data),
                            "examples": [{"id": "", "name": c["name"]} for c in target_client_data[:5]],
                        }
                        canvas_state["target"] = {"state": "ready", "data": target_update_data}
                        yield "canvas_update", {
                            "block": "target",
                            "state": "ready",
                            "data": target_update_data,
                        }

                    if reason_text:
                        canvas_state["reason"] = {"state": "ready", "data": {"text": reason_text}}
                        yield "canvas_update", {
                            "block": "reason",
                            "state": "ready",
                            "data": {"text": reason_text},
                        }

                    message_data: dict[str, Any] = {}
                    if wa_message:
                        message_data["text"] = normalize_campaign_message_template(wa_message)
                    if wa_message_variant:
                        message_data["variant_text"] = normalize_campaign_message_template(wa_message_variant)
                    if treatment_label:
                        message_data["treatment_label"] = treatment_label

                    if message_data:
                        merged_message = {
                            **canvas_state.get("message", {}).get("data", {}),
                            **message_data,
                        }
                        canvas_state["message"] = {"state": "ready", "data": merged_message}
                        yield "canvas_update", {
                            "block": "message",
                            "state": "ready",
                            "data": merged_message,
                        }

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": json.dumps({"status": "ok"}),
                        }
                    )
                    continue

                yield "tool_call", {"tool": tc.name, "params": tc.input}

                result = await execute_tool(tc.name, tc.input, tenant_id)

                if tc.name in target_replace_tools and isinstance(result, list):
                    target_selection_touched = True
                    # Filter to WA-consented clients only for auto-selected targets
                    consented = [c for c in result if c.get("consent_wa")]
                    target_client_data = _dedupe_client_data(consented)
                elif tc.name == "get_clients_reachable_wa" and isinstance(result, list):
                    # Use WA list as fallback target when no service-specific target found yet
                    if not target_client_data:
                        target_client_data = _dedupe_client_data(result)
                        target_selection_touched = True
                elif tc.name == "get_client_by_name" and isinstance(result, list):
                    target_selection_touched = True
                    if not named_lookup_seeded:
                        if named_client_edit_mode == "append":
                            target_client_data = list(existing_target_client_data)
                        elif named_client_edit_mode == "replace":
                            target_client_data = []
                        named_lookup_seeded = True
                    target_client_data = _dedupe_client_data([*target_client_data, *result])

                canvas_update = derive_canvas_update(tc.name, result)
                if canvas_update:
                    existing_block = canvas_state.get(
                        canvas_update["block"],
                        {"state": "empty", "data": {}},
                    )
                    merged_data = {
                        **existing_block.get("data", {}),
                        **canvas_update["data"],
                    }
                    canvas_state[canvas_update["block"]] = {
                        "state": canvas_update["state"],
                        "data": merged_data,
                    }
                    emitted_update = {
                        **canvas_update,
                        "data": merged_data,
                    }
                    yield "canvas_update", emitted_update
                    if canvas_update["block"] == "target":
                        # Use `or` fallback: count=0 (explicit) should fall back to reachable
                        final_target_count = merged_data.get("count") or merged_data.get("reachable", 0)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": json.dumps(result, default=str),
                    }
                )

            claude_messages.append(
                {
                    "role": "assistant",
                    "content": [b.model_dump() for b in response.content],
                }
            )
            claude_messages.append({"role": "user", "content": tool_results})

        if text_parts:
            full_text = "\n".join(text_parts)
            yield "message", {"role": "assistant", "text": full_text}

        if response.stop_reason == "end_turn":
            if "message" not in canvas_state:
                if text_parts:
                    claude_messages.append(
                        {
                            "role": "assistant",
                            "content": [b.model_dump() for b in response.content],
                        }
                    )
                yield "thinking", {"text": "Preparo la proposta strutturata..."}
                forced = await client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1000,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_SCHEMAS,
                    tool_choice={"type": "tool", "name": "propose_campaign"},
                    messages=claude_messages,
                )
                for block in forced.content:
                    if block.type == "tool_use" and block.name == "propose_campaign":
                        objective_summary = (block.input.get("objective_summary") or "").strip()
                        reason_text = (block.input.get("target_reason") or "").strip()
                        wa_message = (block.input.get("wa_message") or "").strip()
                        wa_message_variant = (block.input.get("wa_message_variant") or "").strip()
                        treatment_label = (block.input.get("treatment_label") or "").strip()
                        if objective_summary:
                            objective_data = {
                                **canvas_state.get("objective", {}).get("data", {}),
                                "text": objective_summary,
                            }
                            canvas_state["objective"] = {"state": "ready", "data": objective_data}
                            yield "canvas_update", {
                                "block": "objective",
                                "state": "ready",
                                "data": objective_data,
                            }
                        if reason_text:
                            canvas_state["reason"] = {"state": "ready", "data": {"text": reason_text}}
                            yield "canvas_update", {
                                "block": "reason",
                                "state": "ready",
                                "data": {"text": reason_text},
                            }
                        message_data: dict[str, Any] = {}
                        if wa_message:
                            message_data["text"] = normalize_campaign_message_template(wa_message)
                        if wa_message_variant:
                            message_data["variant_text"] = normalize_campaign_message_template(wa_message_variant)
                        if treatment_label:
                            message_data["treatment_label"] = treatment_label
                        if message_data:
                            canvas_state["message"] = {"state": "ready", "data": message_data}
                            yield "canvas_update", {
                                "block": "message",
                                "state": "ready",
                                "data": message_data,
                            }

            if "objective" in canvas_state:
                canvas_state["objective"]["state"] = "ready"
                yield "canvas_update", {
                    "block": "objective",
                    "state": "ready",
                    "data": canvas_state["objective"]["data"],
                }

            resolved_client_data = (
                target_client_data if target_selection_touched else existing_target_client_data
            )

            if resolved_client_data:
                target_data = {
                    **canvas_state.get("target", {}).get("data", {}),
                    "count": len(resolved_client_data),
                    "reachable": len(resolved_client_data),
                    "examples": [{"id": "", "name": c["name"]} for c in resolved_client_data[:5]],
                }
                canvas_state["target"] = {"state": "ready", "data": target_data}
            elif "target" not in canvas_state:
                canvas_state["target"] = {
                    "state": "ready",
                    "data": {"count": 0, "reachable": 0, "examples": []},
                }

            canvas_state["target"]["state"] = "ready"
            yield "canvas_update", {
                "block": "target",
                "state": "ready",
                "data": canvas_state["target"]["data"],
            }

            if "reason" not in canvas_state:
                canvas_state["reason"] = {
                    "state": "ready",
                    "data": {
                        "text": (
                            "Target definito sui clienti piu coerenti con la richiesta "
                            "e raggiungibili su WhatsApp."
                        )
                    },
                }
                yield "canvas_update", {
                    "block": "reason",
                    "state": "ready",
                    "data": canvas_state["reason"]["data"],
                }

            if "message" not in canvas_state:
                canvas_state["message"] = {
                    "state": "ready",
                    "data": {"text": "Ciao {{nome}}, abbiamo pensato a una proposta dedicata a te."},
                }
                yield "canvas_update", {
                    "block": "message",
                    "state": "ready",
                    "data": canvas_state["message"]["data"],
                }

            creative_data: dict[str, Any] = {}
            treatment_label = (
                canvas_state.get("message", {}).get("data", {}).get("treatment_label")
            )
            if treatment_label:
                creative_data["treatment_label"] = treatment_label
            canvas_state["creative"] = {"state": "ready", "data": creative_data}
            yield "canvas_update", {
                "block": "creative",
                "state": "ready",
                "data": creative_data,
            }

            recipient_count = len(resolved_client_data) if resolved_client_data else final_target_count
            if recipient_count == 0 and initial_target_count > 0 and not target_selection_touched:
                recipient_count = initial_target_count

            excluded_count = canvas_state.get("target", {}).get("data", {}).get("excluded", 0)
            send_data = {
                "recipients": recipient_count,
                "excluded": excluded_count,
                "wa_connected": True,
                "can_send": bool(resolved_client_data or recipient_count > 0),
            }
            canvas_state["send"] = {"state": "ready", "data": send_data}
            yield "canvas_update", {
                "block": "send",
                "state": "ready",
                "data": send_data,
            }

            if campaign_id:
                await _save_campaign(
                    campaign_id,
                    canvas_state,
                    claude_messages,
                    resolved_client_data,
                )

            yield "done", {"campaign_id": campaign_id or ""}
            return

        if response.stop_reason != "tool_use":
            break

    yield "done", {
        "campaign_id": campaign_id or "",
        "aborted": True,
        "reason": "iteration_limit",
    }


async def _save_campaign(
    campaign_id: str,
    canvas_state: dict,
    messages: list,
    client_data: list[dict],
) -> None:
    try:
        sb = get_supabase()
        updates: dict[str, Any] = {
            "status": "ready",
            "chat_history": messages,
        }

        target_summary = dict(canvas_state.get("target", {}).get("data", {}))
        target_summary["client_data"] = client_data
        target_summary["client_phones"] = [c["phone"] for c in client_data]
        updates["target_summary"] = target_summary

        if "objective" in canvas_state:
            updates["objective"] = canvas_state["objective"].get("data", {}).get("text")

        if "message" in canvas_state:
            message_data = canvas_state["message"].get("data", {})
            updates["message_text"] = message_data.get("text")
            updates["message_variant"] = message_data.get("variant_text")
            treatment_label = message_data.get("treatment_label")
            if treatment_label:
                target_summary = dict(updates.get("target_summary") or {})
                target_summary["treatment_label"] = treatment_label
                updates["target_summary"] = target_summary

        if "reason" in canvas_state:
            updates["reason_text"] = canvas_state["reason"].get("data", {}).get("text")

        await asyncio.to_thread(
            lambda: sb.table("wa_campaigns").update(updates).eq("id", campaign_id).execute()
        )
    except Exception as exc:
        logger.error("Failed to save campaign %s: %s", campaign_id, exc)
