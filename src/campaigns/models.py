from typing import Any, Literal, Optional
from pydantic import BaseModel

BlockName = Literal["objective", "target", "reason", "message", "creative", "send"]
BlockState = Literal["empty", "analyzing", "ready", "modified", "approved"]
CampaignStatus = Literal["draft", "analyzing", "ready", "scheduled", "sending", "sent"]


class CanvasBlock(BaseModel):
    state: BlockState = "empty"
    data: dict[str, Any] = {}


class CampaignCanvasState(BaseModel):
    objective: CanvasBlock = CanvasBlock()
    target: CanvasBlock = CanvasBlock()
    reason: CanvasBlock = CanvasBlock()
    message: CanvasBlock = CanvasBlock()
    creative: CanvasBlock = CanvasBlock()
    send: CanvasBlock = CanvasBlock()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class CampaignChatRequest(BaseModel):
    messages: list[ChatMessage]
    campaign_id: Optional[str] = None


class CanvasUpdateEvent(BaseModel):
    block: BlockName
    state: BlockState
    data: dict[str, Any] = {}


class SSEEvent(BaseModel):
    event_type: str
    data: dict[str, Any]
