"""Social Content AI — Entry Point FastAPI."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(".env")
load_dotenv(".env.local")

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SOCIAL")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Social Content AI avviato")

    # Avvia scheduler con job domenicale
    from apscheduler.schedulers.background import BackgroundScheduler
    from src.social.scheduler_jobs import register_social_jobs

    scheduler = BackgroundScheduler(timezone="Europe/Rome")
    register_social_jobs(scheduler)
    scheduler.start()
    logger.info("Scheduler avviato")

    yield

    scheduler.shutdown(wait=False)
    logger.info("Social Content AI fermato")


app = FastAPI(
    title="Social Content AI",
    version="1.0.0",
    description="API per generazione automatica contenuti social per centri estetici",
    lifespan=lifespan,
)

# CORS — permette chiamate da Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registra router social
from src.social.router import router as social_router
app.include_router(social_router)

from src.campaigns.router import router as campaigns_router
app.include_router(campaigns_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "social-content-ai"}
