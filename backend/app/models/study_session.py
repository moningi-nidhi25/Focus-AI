import uuid
from enum import Enum
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from app.utils.database import Base


class SessionStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class StudySession(Base):
    __tablename__ = "study_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    cohort_id = Column(String(36), ForeignKey("cohorts.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String(255), nullable=False)
    status = Column(SQLEnum(SessionStatus), nullable=False, default=SessionStatus.ACTIVE)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_duration_seconds = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    student = relationship("User", back_populates="study_sessions")
    cohort = relationship("Cohort", back_populates="study_sessions")
    study_context = relationship("StudyContext", back_populates="session", uselist=False, cascade="all, delete-orphan")
    content_analyses = relationship("ContentAnalysis", back_populates="session", cascade="all, delete-orphan")
    activity_events = relationship("ActivityEvent", back_populates="session", cascade="all, delete-orphan")
    blocked_contents = relationship("BlockedContent", back_populates="session", cascade="all, delete-orphan")
    focus_metrics = relationship("FocusMetric", back_populates="session", cascade="all, delete-orphan")
    focus_predictions = relationship("FocusPrediction", back_populates="session", cascade="all, delete-orphan")


class StudyContext(Base):
    __tablename__ = "study_contexts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("study_sessions.id", ondelete="CASCADE"), unique=True, nullable=False)
    topic = Column(String(255), nullable=False)
    learning_objectives = Column(Text, nullable=True)
    context_embedding = Column(Text, nullable=True)  # JSON {"model": ..., "vector": [...]} (dense models only)
    course_material = Column(Text, nullable=True)  # optional extracted course notes / documents
    keywords = Column(Text, nullable=True)  # JSON list: structured keywords derived from the context
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("StudySession", back_populates="study_context")

    @property
    def has_course_material(self) -> bool:
        return bool(self.course_material)

    @property
    def has_embedding(self) -> bool:
        return bool(self.context_embedding)
