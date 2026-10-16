import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, field_validator
from app.models.study_session import SessionStatus


class StudyContextBase(BaseModel):
    topic: str
    learning_objectives: Optional[str] = None


class StudyContextCreate(StudyContextBase):
    session_id: Optional[str] = None  # taken from the URL when posting to /sessions/{id}/context
    course_material: Optional[str] = None


class StudyContextRead(StudyContextBase):
    id: str
    session_id: str
    keywords: List[str] = []
    has_course_material: bool = False
    has_embedding: bool = False
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("keywords", mode="before")
    @classmethod
    def _parse_keywords(cls, v):
        if v is None:
            return []
        return json.loads(v) if isinstance(v, str) else v


class StudySessionBase(BaseModel):
    title: str
    cohort_id: Optional[str] = None


class StudySessionCreate(StudySessionBase):
    student_id: str
    topic: str
    learning_objectives: Optional[str] = None
    course_material: Optional[str] = None


class StudySessionUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[SessionStatus] = None
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[int] = None


class StudySessionRead(StudySessionBase):
    id: str
    student_id: str
    status: SessionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: int
    created_at: datetime
    study_context: Optional[StudyContextRead] = None

    model_config = ConfigDict(from_attributes=True)
