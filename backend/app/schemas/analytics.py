from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class FocusMetricBase(BaseModel):
    focus_score: float
    productivity_score: float
    dwell_time_seconds: int = 0
    tab_switch_count: int = 0
    distraction_count: int = 0


class FocusMetricCreate(FocusMetricBase):
    session_id: str


class FocusMetricRead(FocusMetricBase):
    id: str
    session_id: str
    calculated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FocusPredictionBase(BaseModel):
    predicted_focus_score: float
    target_time_window: Optional[str] = None
    confidence: Optional[float] = None
    curve_data: Optional[str] = None


class FocusPredictionCreate(FocusPredictionBase):
    session_id: str


class FocusPredictionRead(FocusPredictionBase):
    id: str
    session_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
