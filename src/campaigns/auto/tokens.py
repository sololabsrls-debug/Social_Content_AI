# src/campaigns/auto/tokens.py
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from src.supabase_client import get_supabase


def generate_token(
    tenant_id: str,
    purpose: str,
    resource_id: str,
    expires_days: int = 7,
) -> str:
    """
    Genera token raw, salva SHA-256 hash in DB. Ritorna token_raw (da mettere nel link).
    purpose: 'monthly_selection' | 'campaign_review'
    """
    sb = get_supabase()
    token_raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token_raw.encode()).hexdigest()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_days)).isoformat()
    sb.table("auto_public_tokens").insert({
        "tenant_id": tenant_id,
        "token_hash": token_hash,
        "purpose": purpose,
        "resource_id": resource_id,
        "expires_at": expires_at,
    }).execute()
    return token_raw


def validate_token(token_raw: str, expected_purpose: str) -> dict:
    """
    Valida token. Ritorna {"tenant_id": ..., "resource_id": ...}.
    Raises HTTPException 401 se invalido, 410 se scaduto.
    """
    sb = get_supabase()
    token_hash = hashlib.sha256(token_raw.encode()).hexdigest()
    res = sb.table("auto_public_tokens").select("*") \
        .eq("token_hash", token_hash).limit(1).execute()
    row = (res.data or [None])[0]
    if not row:
        raise HTTPException(status_code=401, detail="Token non valido")
    if row["purpose"] != expected_purpose:
        raise HTTPException(status_code=401, detail="Token non valido per questo scopo")
    expires = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="Link scaduto")
    if not row.get("used_at"):
        sb.table("auto_public_tokens").update({
            "used_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", row["id"]).execute()
    return {"tenant_id": row["tenant_id"], "resource_id": row["resource_id"]}
