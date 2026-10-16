from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict
from app.schemas.user import UserRead


class CohortBase(BaseModel):
    name: str
    description: Optional[str] = None


class CohortCreate(CohortBase):
    teacher_id: str


class CohortRead(CohortBase):
    id: str
    teacher_id: str
    created_at: datetime
    teacher: Optional[UserRead] = None

    model_config = ConfigDict(from_attributes=True)


class CohortMembershipCreate(BaseModel):
    cohort_id: str
    student_id: str


class CohortMembershipRead(BaseModel):
    cohort_id: str
    student_id: str
    joined_at: datetime
    student: Optional[UserRead] = None

    model_config = ConfigDict(from_attributes=True)
