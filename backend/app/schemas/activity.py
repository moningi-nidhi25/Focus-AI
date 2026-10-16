from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class ActivityEventBase(BaseModel):
    event_type: str
    metadata_json: Optional[str] = None


class ActivityEventCreate(ActivityEventBase):
    session_id: str


class ActivityEventRead(ActivityEventBase):
    id: str
    session_id: str
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class BlockedContentBase(BaseModel):
    url: str
    title: Optional[str] = None
    reason: Optional[str] = None


class BlockedContentCreate(BlockedContentBase):
    session_id: str


class BlockedContentRead(BlockedContentBase):
    id: str
    session_id: str
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
