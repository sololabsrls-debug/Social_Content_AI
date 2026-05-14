# src/campaigns/auto/models.py
from pydantic import BaseModel
from typing import Optional


class AutoCampaignConfigIn(BaseModel):
    campaigns_per_month: int
    is_active: bool


class AutoBundleOrderIn(BaseModel):
    sort_order: int


class AutoCampaignRescheduleIn(BaseModel):
    scheduled_at: str  # ISO-8601 UTC datetime string
