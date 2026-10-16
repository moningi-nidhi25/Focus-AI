import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.utils.database import Base


class FocusMetric(Base):
    __tablename__ = "focus_metrics"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("study_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    focus_score = Column(Float, nullable=False)
    productivity_score = Column(Float, nullable=False)
    dwell_time_seconds = Column(Integer, default=0)
    tab_switch_count = Column(Integer, default=0)
    distraction_count = Column(Integer, default=0)
    calculated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("StudySession", back_populates="focus_metrics")


class FocusPrediction(Base):
    __tablename__ = "focus_predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("study_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    predicted_focus_score = Column(Float, nullable=False)
    target_time_window = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    curve_data = Column(Text, nullable=True)  # JSON string representing focus decay curve points
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("StudySession", back_populates="focus_predictions")
