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

from src.campaigns.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
from src.supabase_client import get_supabase

logger = logging.getLogger("CAMPAIGNS.agent")

SYSTEM_PROMPT = """Sei un assistente marketing esperto per centri estetici. Aiuti l'estetista a creare campagne WhatsApp efficaci e mirate.

Quando ricevi un obiettivo di marketing:
1. Usa i tool disponibili per analizzare i dati reali del centro (clienti, appuntamenti, servizi)
2. Identifica il target giusto con motivazioni concrete
3. Proponi un messaggio WhatsApp persuasivo e personalizzato con {{nome}} come variabile
4. Spiega le tue scelte in modo semplice, diretto e non tecnico

Regole importanti:
- Parla sempre in italiano
- Sii diretto e concreto come un marketing manager senior
- Non suggerire mai l'invio senza che l'estetista approvi esplicitamente
- Fai poche domande, solo se strettamente necessarie per migliorare la campagna
- Usa un tono caldo e professionale, mai freddo o robotico
- Non usare trattini come separatori nei messaggi
- Il messaggio WhatsApp proposto deve essere pronto per l'invio, non un template astratto

Formato output obbligatorio quando proponi un messaggio WhatsApp:
- Prima spiega brevemente il ragionamento (2-3 frasi)
- Poi scrivi esattamente "MESSAGGIO:" su una riga separata
- Poi scrivi il messaggio WhatsApp (con {{nome}} come segnaposto per il nome cliente)
- Poi chiedi se vuole modificare tono o contenuto

Esempio:
Ho trovato 12 clienti che non vengono da più di 3 mesi, tutte raggiungibili su WhatsApp. Questa campagna punta sulla nostalgia e un piccolo vantaggio esclusivo per invogliarle a tornare.

MESSAGGIO:
Ciao {{nome}}! 😊 Ci manchi tantissimo al nostro centro. Vieni a trovarci questa settimana e ti riserviamo uno sconto speciale del 15% su qualsiasi trattamento. Ti aspettiamo! 💆‍♀️"""


def derive_canvas_update(tool_name: str, result: Any) -> Optional[dict]:
    """
    Derives a canvas block update from a tool result.
    Returns None if the tool result doesn't map to a canvas block.
    """
    client_tools = {
        "get_clients_by_service",
        "get_inactive_clients",
        "get_clients_reachable_wa",
        "get_clients_by_ltv",
        "get_clients_with_birthday",
        "get_clients_by_tag",
        "get_clients_never_returned",
    }

    if tool_name in client_tools and isinstance(result, list):
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


def _extract_message_and_reason(full_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Splits Claude's response into (reason_text, wa_message_text).
    Looks for the 'MESSAGGIO:' marker. Falls back to {{nome}} detection.
    """
    if "MESSAGGIO:" in full_text:
        parts = full_text.split("MESSAGGIO:", 1)
        reason = parts[0].strip() or None
        # The WA message is everything after the marker until the next blank-line paragraph
        after_marker = parts[1].strip()
        # Take until we hit a blank line that signals the post-message commentary
        wa_lines = []
        for line in after_marker.splitlines():
            if not line.strip() and wa_lines:
                break
            wa_lines.append(line)
        wa_message = "\n".join(wa_lines).strip() or None
        return reason, wa_message

    # Fallback: if {{nome}} appears anywhere, treat whole text as message (old behavior)
    if "{{nome}}" in full_text:
        return None, full_text

    return None, None


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
) -> AsyncGenerator[tuple[str, dict], None]:
    """
    Runs the Claude tool_use loop and yields (event_type, data) tuples.
    Event types: thinking | tool_call | canvas_update | message | done
    """
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    claude_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    canvas_state: dict[str, Any] = {}
    final_target_count = 0

    # Emit objective from the last user message immediately
    last_user_text = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    if last_user_text:
        canvas_state["objective"] = {"state": "analyzing", "data": {"text": last_user_text}}
        yield "canvas_update", {
            "block": "objective",
            "state": "analyzing",
            "data": {"text": last_user_text},
        }

    for _ in range(10):  # safety limit on tool loops
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
                yield "tool_call", {"tool": tc.name, "params": tc.input}

                result = await execute_tool(tc.name, tc.input, tenant_id)

                canvas_update = derive_canvas_update(tc.name, result)
                if canvas_update:
                    canvas_state[canvas_update["block"]] = {
                        "state": canvas_update["state"],
                        "data": canvas_update["data"],
                    }
                    yield "canvas_update", canvas_update
                    if canvas_update["block"] == "target":
                        final_target_count = canvas_update["data"].get(
                            "count", canvas_update["data"].get("reachable", 0)
                        )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

            claude_messages.append({
                "role": "assistant",
                "content": [b.model_dump() for b in response.content],
            })
            claude_messages.append({"role": "user", "content": tool_results})

        if text_parts:
            full_text = "\n".join(text_parts)
            yield "message", {"role": "assistant", "text": full_text}

            reason_text, wa_message = _extract_message_and_reason(full_text)

            if reason_text:
                canvas_state["reason"] = {"state": "ready", "data": {"text": reason_text}}
                yield "canvas_update", {
                    "block": "reason",
                    "state": "ready",
                    "data": {"text": reason_text},
                }

            if wa_message:
                canvas_state["message"] = {"state": "ready", "data": {"text": wa_message}}
                yield "canvas_update", {
                    "block": "message",
                    "state": "ready",
                    "data": {"text": wa_message},
                }

        if response.stop_reason == "end_turn":
            if "target" in canvas_state:
                canvas_state["target"]["state"] = "ready"
                yield "canvas_update", {
                    "block": "target",
                    "state": "ready",
                    "data": canvas_state["target"]["data"],
                }

            if final_target_count > 0:
                canvas_state["send"] = {
                    "state": "ready",
                    "data": {"recipients": final_target_count},
                }
                yield "canvas_update", {
                    "block": "send",
                    "state": "ready",
                    "data": {"recipients": final_target_count},
                }

            # Mark objective as ready
            if "objective" in canvas_state:
                canvas_state["objective"]["state"] = "ready"
                yield "canvas_update", {
                    "block": "objective",
                    "state": "ready",
                    "data": canvas_state["objective"]["data"],
                }

            if campaign_id:
                await _save_campaign(campaign_id, canvas_state, claude_messages)

            yield "done", {"campaign_id": campaign_id or ""}
            return

        if response.stop_reason != "tool_use":
            break

    yield "done", {"campaign_id": campaign_id or "", "aborted": True, "reason": "iteration_limit"}


async def _save_campaign(campaign_id: str, canvas_state: dict, messages: list) -> None:
    try:
        sb = get_supabase()
        updates: dict[str, Any] = {
            "status": "ready",
            "chat_history": messages,
        }
        if "target" in canvas_state:
            updates["target_summary"] = canvas_state["target"].get("data")
        if "message" in canvas_state:
            updates["message_text"] = canvas_state["message"].get("data", {}).get("text")
        if "reason" in canvas_state:
            updates["reason_text"] = canvas_state["reason"].get("data", {}).get("text")
        await asyncio.to_thread(
            lambda: sb.table("wa_campaigns").update(updates).eq("id", campaign_id).execute()
        )
    except Exception as exc:
        logger.error("Failed to save campaign %s: %s", campaign_id, exc)
