from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field

EventType = Literal[
    "session_start", "session_end", "page_open", "page_close", "tab_switch", "click",
    "scroll", "dwell_time", "content_allowed", "warning_shown", "content_blocked",
    "distraction_attempt",
]


class TrackedEvent(BaseModel):
    event_type: EventType
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EventBatch(BaseModel):
    session_id: str
    events: List[TrackedEvent] = Field(min_length=1, max_length=200)


class EventBatchResult(BaseModel):
    stored: int
    snapshot_created: bool


class MetricsSummary(BaseModel):
    session_id: str
    study_duration_seconds: int
    total_dwell_seconds: int
    relevant_seconds: int
    allowed_count: int
    warning_count: int
    blocked_count: int
    distraction_count: int
    tab_switch_count: int
    click_count: int
    scroll_count: int
    focus_score: float
    productivity_score: float
    insights: List[str]