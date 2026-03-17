"""Job schedulati — pipeline domenicale."""

import asyncio
import logging

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger("SOCIAL.scheduler")

try:
    import sentry_sdk
    _SENTRY = True
except ImportError:
    _SENTRY = False


def _run_async(coro):
    """Esegue coroutine async da contesto sync (APScheduler)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(coro)
        else:
            loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)


def _job_weekly_social():
    """Domenica 18:00 Roma: genera piano contenuti settimana successiva."""
    logger.info("Job weekly_social avviato")
    try:
        from src.social.content_pipeline import run_all_tenants
        _run_async(run_all_tenants())
    except Exception as e:
        logger.exception(f"Errore job weekly_social: {e}")
        if _SENTRY:
            sentry_sdk.capture_exception(e)


def register_social_jobs(scheduler: BackgroundScheduler) -> None:
    """Aggiunge i job social allo scheduler esistente."""
    scheduler.add_job(
        _job_weekly_social,
        "cron",
        day_of_week="sun",
        hour=18,
        minute=0,
        id="weekly_social_content",
        name="Weekly social content generation",
    )
    logger.info("Job social registrato: weekly_social_content (domenica 18:00)")
