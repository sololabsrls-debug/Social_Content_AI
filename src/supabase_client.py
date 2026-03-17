"""Supabase client singleton — service_role bypassa RLS."""

import os
import logging
from supabase import create_client, Client

logger = logging.getLogger("SOCIAL.supabase")
_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY devono essere impostati")
        _client = create_client(url, key)
        logger.info("Supabase client inizializzato")
    return _client
