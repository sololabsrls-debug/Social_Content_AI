# src/campaigns/auto/models.py
from pydantic import BaseModel
from typing import Optional, List


class AutoCampaignConfigIn(BaseModel):
    campaigns_per_month: int
    is_active: bool


class AutoBundleOrderIn(BaseModel):
    sort_order: int


class AutoCampaignRescheduleIn(BaseModel):
    scheduled_at: str  # ISO-8601 UTC datetime string


VALID_PROMO_TYPES = {
    "service_product", "service_only", "product_only",
    "multi_session", "service_service", "product_bundle",
    "seasonal", "reactivation",
}


class ProposalSelectIn(BaseModel):
    proposal_ids: List[str]


class PublicMessageUpdateIn(BaseModel):
    message_text: str


class PublicTargetUpdateIn(BaseModel):
    target_summary: str


class PublicImageUploadIn(BaseModel):
    image_data: str   # base64
    mime_type: str    # es. "image/jpeg"
