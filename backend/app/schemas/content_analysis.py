from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict
from app.models.content_analysis import DecisionEnum


class ContentAnalysisBase(BaseModel):
    url: str
    title: Optional[str] = None
    extracted_text: Optional[str] = None
    relevance_score: float
    decision: DecisionEnum
    explanation: Optional[str] = None
    key_concepts: Optional[str] = None
    classification: Optional[str] = None
    heatmap: Optional[str] = None
    signals: Optional[str] = None


class ContentAnalysisCreate(ContentAnalysisBase):
    session_id: str


class ContentAnalysisRead(ContentAnalysisBase):
    id: str
    session_id: str
    analyzed_at: datetime

    model_config = ConfigDict(from_attributes=True)
