from typing import List, Optional
from pydantic import BaseModel


class CurvePoint(BaseModel):
    window: int          # 1 = next window, 2 = the one after, ...
    predicted: float
    lower: float
    upper: float


class PredictionResult(BaseModel):
    session_id: str
    current_focus: float
    curve: List[CurvePoint]
    alert: bool
    alert_window: Optional[int]
    message: str
    history_points: int
    model_note: str